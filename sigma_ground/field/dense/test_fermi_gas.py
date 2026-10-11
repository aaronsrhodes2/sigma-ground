"""field.dense.fermi_gas: closed forms against direct quadrature, limits, thermodynamic identities, finite-T limits.

Ground truth is independent of the code: scipy quadrature of the defining integrals, the NR/UR asymptotes, the Gibbs-Duhem relation
(P = n mu - eps), and the classical and Sommerfeld limits. Labels (BUG, BY CONSTRUCTION, DATA, SOFT) follow misc/NUCLEAR_MATTER_PLAN.md.
"""
from __future__ import annotations

import math
import unittest

from scipy.integrate import quad

from sigma_ground.field.constants import AMU_KG, C, K_B
from sigma_ground.field.dense import fermi_gas as fg
from sigma_ground.field.dense.fermi_gas import ELECTRON, NEUTRON


class TestClosedForms(unittest.TestCase):
    def test_pressure_energy_trace_match_direct_quadrature(self):             # BUG
        for x in (1e-3, 0.05, 0.299, 0.301, 0.8, 1.0, 5.0, 40.0):
            p = quad(lambda t: t ** 4 / math.sqrt(1 + t * t), 0, x, epsabs=0, epsrel=1e-13)[0] / 3.0
            ek = quad(lambda t: t * t * (math.sqrt(1 + t * t) - 1.0), 0, x, epsabs=0, epsrel=1e-13)[0]
            tr = quad(lambda t: t * t / math.sqrt(1 + t * t), 0, x, epsabs=0, epsrel=1e-13)[0]
            self.assertAlmostEqual(fg._i_pressure(x) / p, 1.0, delta=1e-10, msg=f"P x={x}")
            self.assertAlmostEqual(fg._i_energy_kinetic(x) / ek, 1.0, delta=1e-9, msg=f"E x={x}")
            self.assertAlmostEqual(fg._i_trace(x) / tr, 1.0, delta=1e-10, msg=f"trace x={x}")

    def test_series_and_closed_form_agree_across_the_switch(self):            # BUG
        a, b = fg._i_pressure(fg._SERIES_X - 1e-12), fg._i_pressure(fg._SERIES_X + 1e-12)
        self.assertAlmostEqual(a / b, 1.0, delta=1e-9)

    def test_non_relativistic_limit(self):                                    # BUG
        self.assertAlmostEqual(fg._i_pressure(1e-3) / (1e-3 ** 5 / 15), 1.0, delta=1e-5)

    def test_ultra_relativistic_limit(self):                                  # BUG
        x = 1e4
        self.assertAlmostEqual(fg._i_pressure(x) / (x ** 4 / 12), 1.0, delta=1e-3)
        eps = fg._i_energy_kinetic(x) + x ** 3 / 3
        self.assertLess(3 * fg._i_pressure(x) / eps, 1.0)
        self.assertAlmostEqual(3 * fg._i_pressure(x) / eps, 1.0, delta=1e-3)

    def test_free_gas_trace_is_positive(self):                                # BY CONSTRUCTION (theorem; matters for the trace diagnostic)
        for x in (1e-6, 1e-2, 0.5, 3.0, 1e3):
            self.assertGreater(fg._i_trace(x), 0.0)


class TestThermodynamics(unittest.TestCase):
    def test_gibbs_duhem(self):                                               # BUG: P = n (mu - m c^2) - eps_kin, an identity
        for n in (1e30, 1e33, 1e36, 1e44):
            s = fg.eos_T0(n, NEUTRON)
            mu_kin = NEUTRON.mc2_j * s.x * s.x / (math.sqrt(1 + s.x * s.x) + 1.0)        # mu - m c^2 without cancellation
            self.assertAlmostEqual((n * mu_kin - s.kinetic_energy_density_jm3) / s.pressure_pa, 1.0, delta=1e-9, msg=f"n={n}")

    def test_first_law_self_consistency(self):                                # BUG: d eps / dn = (eps + P)/n by finite difference
        n = 3e44
        h = n * 1e-5
        d_eps = (fg.eos_T0(n + h, NEUTRON).energy_density_jm3 - fg.eos_T0(n - h, NEUTRON).energy_density_jm3) / (2 * h)
        s = fg.eos_T0(n, NEUTRON)
        self.assertAlmostEqual(d_eps * n / (s.energy_density_jm3 + s.pressure_pa), 1.0, delta=1e-7)

    def test_causality_and_sound_speed(self):                                 # BUG: c_s^2 from finite differences, not from the code's formula
        for n in (1e32, 1e40, 1e46):
            h = n * 1e-5
            a, b = fg.eos_T0(n - h, NEUTRON), fg.eos_T0(n + h, NEUTRON)
            cs2 = (b.pressure_pa - a.pressure_pa) / (b.energy_density_jm3 - a.energy_density_jm3)      # dimensionless: c_s^2 / c^2
            self.assertGreater(cs2, 0.0)
            self.assertLess(cs2, 1.0 / 3.0)
            self.assertAlmostEqual(cs2 / fg.eos_T0(n, NEUTRON).sound_speed_sq_over_c2, 1.0, delta=1e-5)

    def test_white_dwarf_pressure_share(self):                                # BUG: 3P/(rho c^2) at 1e9 kg/m^3, mu_e = 2 (plan: 8.77e-5, x = 0.8007)
        s = fg.electron_gas(1e9, 2.0)
        self.assertAlmostEqual(s.x, 0.8007, delta=2e-4)
        self.assertAlmostEqual(3 * s.pressure_pa / (1e9 * C ** 2), 8.77e-5, delta=0.02e-5)

    def test_chandrasekhar_pressure_at_1e9(self):                             # DATA (cross-check vs the plan's figure): 2.63e21 Pa at 1e9 kg/m^3
        self.assertAlmostEqual(fg.electron_gas(1e9, 2.0).pressure_pa / 2.63e21, 1.0, delta=0.01)

    def test_electron_gas_density_to_x_is_the_textbook_formula(self):         # BUG: x = (3 pi^2 n_e)^(1/3) hbar / (m_e c), n_e = rho/(mu_e m_u)
        from sigma_ground.field.constants import HBAR, M_ELECTRON_KG
        rho = 3.2e8
        n_e = rho / (2.0 * AMU_KG)
        self.assertAlmostEqual(fg.electron_gas(rho, 2.0).x, HBAR * (3 * math.pi ** 2 * n_e) ** (1 / 3) / (M_ELECTRON_KG * C), delta=1e-12)

    def test_neutron_gas_x_equals_one_at_the_scale_density(self):             # BUG
        self.assertAlmostEqual(fg.relativity_parameter(NEUTRON.n0_m3, NEUTRON), 1.0, delta=1e-12)


class TestFiniteTemperature(unittest.TestCase):
    def test_classical_limit_is_n_k_T(self):                                  # BUG
        n, T = 1e22, 1e6                                                      # T/T_F ~ 1e4 for electrons here
        s = fg.eos_finite_T_nonrel(n, T, ELECTRON)
        self.assertAlmostEqual(s.pressure_pa / (n * K_B * T), 1.0, delta=1e-3)

    def test_sommerfeld_low_temperature_correction(self):                     # BUG: P = P0 [1 + (5 pi^2 / 12)(kT/E_F)^2]
        n = 1e30
        p0 = fg.eos_T0(n, ELECTRON).pressure_pa
        e_f = fg.fermi_momentum(n, ELECTRON) ** 2 / (2 * ELECTRON.mass_kg)
        T = 0.02 * e_f / K_B
        s = fg.eos_finite_T_nonrel(n, T, ELECTRON)
        pred = 1.0 + (5 * math.pi ** 2 / 12) * (K_B * T / e_f) ** 2
        # T = 0 form includes the relativistic correction at x ~ 0.07 (~1e-3); compare the non-relativistic pressure instead
        p0_nr = 0.4 * n * e_f
        self.assertAlmostEqual(s.pressure_pa / p0_nr, pred, delta=2e-3)

    def test_zero_temperature_falls_back_to_exact(self):                      # BY CONSTRUCTION
        self.assertEqual(fg.eos_finite_T_nonrel(1e30, 0.0, ELECTRON).pressure_pa, fg.eos_T0(1e30, ELECTRON).pressure_pa)


class TestRegimes(unittest.TestCase):
    def test_labels(self):                                                    # BY CONSTRUCTION
        self.assertEqual(fg.electron_gas(1e9, 2.0).regime, "SAFE")
        self.assertEqual(fg.electron_gas(1e3, 2.0).regime, "EDGE")            # x < 0.1: Coulomb corrections not modelled here
        self.assertEqual(fg.eos_finite_T_nonrel(1e30, 1e9, ELECTRON).regime, "BEYOND" if fg.relativity_parameter(1e30, ELECTRON) > 0.3 else "SAFE")

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            fg.electron_gas(-1.0)
        with self.assertRaises(ValueError):
            fg.fermi_momentum(-1.0)


if __name__ == "__main__":
    unittest.main()
