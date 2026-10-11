"""field.dense.nuclear_matter: SLy4 saturation properties, thermodynamic identities, beta equilibrium, the maximum mass, and the real-data gates.

Labels follow misc/NUCLEAR_MATTER_PLAN.md:  BUG = identity or limit,  LIT = reproduces a published calculation (a failure is a code or transcription
error, not evidence),  DATA = compares with a measurement (a failure is a physics statement about the EOS, not a bug).
Radii are NOT scored: the EOS is core-only (no crust), which shifts radii by ~0.5-1 km; the maximum mass is insensitive to it.
"""
from __future__ import annotations

import math
import unittest

from sigma_ground.field.constants import M_SUN_KG
from sigma_ground.field.dense import eos as eosmod
from sigma_ground.field.dense import fermi_gas as fg
from sigma_ground.field.dense import nuclear_matter as nm
from sigma_ground.field.dense.tov import maximum_mass, solve_tov

_EOS = nm.npe_mu_eos()
_LO, _HI = _EOS.pressure_range_pa
_MAX = maximum_mass(_EOS, _LO * 10, _HI * 0.95)


class TestSaturationProperties(unittest.TestCase):
    def test_sly4_reproduces_its_published_saturation_point(self):            # LIT: Chabanat et al. 1998 (guards the parameter transcription)
        s = nm.SLY4.saturation_properties()
        self.assertAlmostEqual(s["n0_fm3"], 0.1595, delta=5e-4)
        self.assertAlmostEqual(s["E_over_A_MeV"], -15.97, delta=0.02)
        self.assertAlmostEqual(s["K_MeV"], 229.9, delta=0.3)
        self.assertAlmostEqual(s["J_MeV"], 32.0, delta=0.1)
        self.assertAlmostEqual(s["L_MeV"], 45.9, delta=0.2)

    def test_zero_pressure_at_saturation(self):                              # BUG: P(n_0, delta = 0) = 0
        n0 = nm.SLY4.saturation_properties()["n0_fm3"]
        self.assertLess(abs(nm.SLY4.pressure(n0 / 2, n0 / 2)), 1e-8)         # MeV/fm^3

    def test_a_wrong_parameter_breaks_the_check(self):                       # BY CONSTRUCTION: the check can fail
        bad_t3 = nm.Skyrme("bad", t0=-2488.91, t1=486.82, t2=-546.39, t3=13000.0, x0=0.834, x1=-0.344, x2=-1.0, x3=1.354, sigma=1.0 / 6.0)
        self.assertGreater(abs(bad_t3.saturation_properties()["E_over_A_MeV"] + 15.97), 0.5)
        # x3 drops out of symmetric matter (its bracket is 3/2 for any x3), so it is checked through the symmetry energy instead
        bad_x3 = nm.Skyrme("bad", t0=-2488.91, t1=486.82, t2=-546.39, t3=13777.0, x0=0.834, x1=-0.344, x2=-1.0, x3=1.0, sigma=1.0 / 6.0)
        self.assertGreater(abs(bad_x3.saturation_properties()["J_MeV"] - 32.0), 0.5)

    def test_pressure_is_n_squared_times_the_energy_per_baryon_slope(self):  # BUG: independent of the mu-based pressure formula
        for n, d in ((0.1, 0.0), (0.16, 0.0), (0.3, 0.4), (0.8, 0.8)):
            h = n * 1e-5
            slope = (nm.SLY4.energy_per_baryon(n + h, d) - nm.SLY4.energy_per_baryon(n - h, d)) / (2 * h)
            nn, npr = 0.5 * n * (1 + d), 0.5 * n * (1 - d)
            self.assertAlmostEqual(nm.SLY4.pressure(nn, npr) / (n * n * slope), 1.0, delta=1e-6, msg=f"n={n} d={d}")

    def test_analytic_chemical_potentials_match_finite_differences(self):   # BUG: two independent evaluations of d eps / d n_q
        for nn, npr in ((0.08, 0.08), (0.2, 0.01), (0.5, 0.05), (1.0, 0.3)):
            a, b = nm.SLY4.mu(nn, npr), nm.SLY4.mu_numeric(nn, npr)
            self.assertAlmostEqual(a[0] / b[0], 1.0, delta=1e-8)
            self.assertAlmostEqual(a[1] / b[1], 1.0, delta=1e-8)

    def test_pure_neutron_matter_is_unbound(self):                           # DATA-like: neutron matter has E/A > 0 at all densities in SLy4
        for n in (0.05, 0.16, 0.5):
            self.assertGreater(nm.SLY4.energy_per_baryon(n, 1.0), 0.0)


class TestBetaEquilibrium(unittest.TestCase):
    def test_charge_neutrality_and_chemical_equilibrium(self):               # BUG
        for nb in (0.1, 0.2, 0.5, 1.0):
            b = nm.beta_equilibrium(nb)
            self.assertAlmostEqual(b.n_p - b.n_e - b.n_mu, 0.0, delta=1e-12)
            self.assertAlmostEqual(b.n_n + b.n_p, nb, delta=1e-12)
            mun, mup = nm.SLY4.mu(b.n_n, b.n_p)
            mue = mun - mup
            x_e = (b.n_e * 1e45 / fg.ELECTRON.n0_m3) ** (1 / 3)
            self.assertAlmostEqual(nm._MC2_E * math.sqrt(1 + x_e * x_e), mue, delta=1e-6)

    def test_muons_appear_when_mu_e_exceeds_the_muon_mass(self):             # BUG: onset at mu_e = m_mu c^2
        b_low, b_high = nm.beta_equilibrium(0.08), nm.beta_equilibrium(0.8)
        self.assertEqual(b_low.n_mu, 0.0)
        self.assertGreater(b_high.n_mu, 0.0)
        mun, mup = nm.SLY4.mu(b_high.n_n, b_high.n_p)
        self.assertGreater(mun - mup, nm._MC2_MU)

    def test_proton_fraction_is_small_and_rises_with_density(self):          # DATA-like: Y_p ~ 0.04-0.1 near n_0 in SLy4 beta matter
        y = [nm.beta_equilibrium(n).n_p / n for n in (0.1, 0.16, 0.3, 0.6)]
        self.assertTrue(all(b > a for a, b in zip(y, y[1:])))
        self.assertTrue(0.02 < y[1] < 0.15)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            nm.beta_equilibrium(-0.1)


class TestStarsFromTheEOS(unittest.TestCase):
    def test_first_law_on_the_table(self):                                    # BUG: d eps / dn = (eps + P)/n
        for f in (3.0, 30.0, 300.0):
            self.assertLess(eosmod.first_law_error(_EOS, _LO * f), 1e-5)

    def test_douchin_haensel_maximum_mass(self):                              # LIT: DH2001 SLy M_max = 2.05 M_sun (a failure = a bug in TOV, EOS or units)
        self.assertAlmostEqual(_MAX.mass_msun, 2.05, delta=0.05)
        self.assertLess(_MAX.compactness, 4.0 / 9.0)

    def test_causality_up_to_the_maximum_mass_star(self):                     # BUG (regime WALL otherwise): c_s^2 <= c^2 below the central pressure
        for f in (0.02, 0.1, 0.3, 0.6, 0.95):
            p = _MAX.central_pressure_pa * f
            h = p * 1e-4
            cs2 = (_EOS.eps(p + h) - _EOS.eps(p - h))
            cs2 = 2 * h / cs2
            self.assertTrue(0.0 < cs2 <= 1.0, msg=f"P/Pc={f}: cs2={cs2}")


class TestRealDataGates(unittest.TestCase):
    """The EOS is scored against measurements. Each failure would be a statement about the EOS, not about the code."""

    def test_gw170817_pressure_window_at_twice_nuclear_density(self):        # DATA: Abbott et al. 2018: P(2 rho_nuc) in [1.8, 6.2]e33 Pa at 90 %, rho_nuc = 2.8e14 g/cm^3
        rho = 2.0 * 2.8e17                                                    # kg/m^3
        n_b = rho / 1.66053906660e-27 / 1e45                                  # fm^-3, rest-mass density with the atomic mass unit
        p_pa = nm.beta_equilibrium(n_b).pressure_mev_fm3 * nm._MEVFM3_TO_JM3
        self.assertTrue(1.8e33 <= p_pa <= 6.2e33, msg=f"P = {p_pa:.3e} Pa")

    def test_the_gw170817_window_is_weak_it_does_not_exclude_the_free_gas(self):   # DATA (a measured fact about the gate, not a pass for the EOS)
        # The free neutron gas gives 1.94e33 Pa at 2 rho_nuc, inside [1.8, 6.2]e33: this gate alone cannot tell a realistic EOS from a free gas,
        # which is why the 2 solar-mass floor below is the discriminating test.
        rho = 2.0 * 2.8e17
        n = rho / 1.66053906660e-27
        x = fg.relativity_parameter(n, fg.NEUTRON)
        p_pa = fg.chandrasekhar_pressure(x, fg.NEUTRON)
        self.assertTrue(1.8e33 <= p_pa <= 6.2e33, msg=f"free gas P = {p_pa:.3e} Pa")

    def test_two_solar_mass_floor(self):                                      # DATA: PSR J0740+6620, M = 2.08 +/- 0.07 M_sun (Fonseca 2021): M_max >= 1.94 (2 sigma)
        self.assertGreaterEqual(_MAX.mass_msun, 1.94)

    def test_free_neutron_star_fails_the_two_solar_mass_floor(self):         # DATA (negative control)
        free = eosmod.free_fermi_eos(fg.NEUTRON)
        m = maximum_mass(free, fg.chandrasekhar_pressure(0.3, fg.NEUTRON), fg.chandrasekhar_pressure(3.0, fg.NEUTRON))
        self.assertLess(m.mass_msun, 1.94)


if __name__ == "__main__":
    unittest.main()
