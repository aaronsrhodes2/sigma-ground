"""
Tests for composites.py — effective properties of multi-phase materials.

Strategy:
  - Test Voigt/Reuss bounds are mathematically correct
  - Test Halpin-Tsai recovers Voigt at ξ→∞ and Reuss at ξ→0
  - Test Hashin-Shtrikman bounds are tighter than Voigt-Reuss
  - Test Gibson-Ashby gives reasonable foam properties
  - Test named composites against MEASURED engineering data
  - Test Rule 9: every composite in COMPOSITES gets a report

Reference values (MEASURED, from composite engineering handbooks):
  CFRP unidirectional: E_∥ ≈ 130-150 GPa, ρ ≈ 1560 kg/m³
  GFRP unidirectional: E_∥ ≈ 38-45 GPa, ρ ≈ 1940 kg/m³
  Concrete: E ≈ 25-40 GPa, ρ ≈ 2300-2400 kg/m³
  WC-Co: E ≈ 550-650 GPa, ρ ≈ 14800 kg/m³
"""

import math
import unittest

from sigma_ground.field.interface.composites import (
    voigt_bound,
    reuss_bound,
    voigt_reuss_hill,
    hashin_shtrikman_bounds,
    halpin_tsai,
    gibson_ashby_modulus,
    gibson_ashby_strength,
    density_rule_of_mixtures,
    thermal_expansion_voigt,
    thermal_conductivity_bounds,
    composite_modulus,
    composite_density,
    composite_expansion,
    specific_stiffness,
    composite_report,
    full_report,
    COMPOSITES,
)


class TestVoigtReuss(unittest.TestCase):
    """Fundamental bounds on composite modulus."""

    def test_voigt_is_upper_bound(self):
        """Voigt ≥ Reuss for any positive moduli and fractions."""
        E = [100e9, 10e9]
        for f in [0.1, 0.3, 0.5, 0.7, 0.9]:
            frac = [f, 1 - f]
            with self.subTest(f=f):
                self.assertGreaterEqual(
                    voigt_bound(E, frac),
                    reuss_bound(E, frac)
                )

    def test_voigt_at_f1_equals_E1(self):
        """100% phase 1 → E = E₁."""
        self.assertAlmostEqual(
            voigt_bound([100, 10], [1.0, 0.0]), 100, places=5
        )

    def test_reuss_at_f1_equals_E1(self):
        """100% phase 1 → E = E₁."""
        self.assertAlmostEqual(
            reuss_bound([100, 10], [1.0, 0.0]), 100, places=5
        )

    def test_vrh_between_bounds(self):
        """VRH average should be between Voigt and Reuss."""
        E = [200e9, 5e9]
        frac = [0.4, 0.6]
        vrh = voigt_reuss_hill(E, frac)
        self.assertGreaterEqual(vrh, reuss_bound(E, frac))
        self.assertLessEqual(vrh, voigt_bound(E, frac))

    def test_three_phases(self):
        """Works with 3+ phases."""
        E = [100, 50, 10]
        f = [0.3, 0.4, 0.3]
        v = voigt_bound(E, f)
        r = reuss_bound(E, f)
        self.assertGreater(v, r)
        self.assertAlmostEqual(v, 30 + 20 + 3, places=5)


class TestHashinShtrikman(unittest.TestCase):
    """HS bounds are tighter than Voigt-Reuss."""

    def test_hs_tighter_than_voigt_reuss(self):
        """HS bounds should be inside Voigt-Reuss bounds."""
        K1, G1 = 50e9, 30e9    # soft phase
        K2, G2 = 200e9, 80e9   # stiff phase
        f1 = 0.4

        K_hs_lo, K_hs_hi = hashin_shtrikman_bounds(K1, G1, f1, K2, G2)
        K_voigt = voigt_bound([K1, K2], [f1, 1 - f1])
        K_reuss = reuss_bound([K1, K2], [f1, 1 - f1])

        self.assertGreaterEqual(K_hs_lo, K_reuss - 1)  # small tolerance
        self.assertLessEqual(K_hs_hi, K_voigt + 1)
        self.assertLessEqual(K_hs_lo, K_hs_hi)

    def test_hs_symmetric_at_equal_fractions(self):
        """At f1 = 0.5 with equal phases, bounds should converge."""
        K, G = 100e9, 50e9
        K_lo, K_hi = hashin_shtrikman_bounds(K, G, 0.5, K, G)
        self.assertAlmostEqual(K_lo, K_hi, delta=K * 0.01)


class TestHalpinTsai(unittest.TestCase):
    """Halpin-Tsai model for fiber/particle composites."""

    def test_recovers_voigt_at_large_xi(self):
        """ξ → ∞ should give Voigt bound (continuous fibers)."""
        E_f, E_m = 230e9, 3.5e9
        f = 0.6
        HT = halpin_tsai(E_f, E_m, f, xi=1e6)
        V = voigt_bound([E_f, E_m], [f, 1 - f])
        self.assertAlmostEqual(HT / V, 1.0, delta=0.05)

    def test_between_bounds(self):
        """HT should be between Voigt and Reuss."""
        E_f, E_m = 100e9, 3e9
        f = 0.3
        for xi in [0.5, 2, 10, 50]:
            with self.subTest(xi=xi):
                HT = halpin_tsai(E_f, E_m, f, xi)
                V = voigt_bound([E_f, E_m], [f, 1 - f])
                R = reuss_bound([E_f, E_m], [f, 1 - f])
                self.assertGreaterEqual(HT, R * 0.99)
                self.assertLessEqual(HT, V * 1.01)

    def test_zero_filler_gives_matrix(self):
        """f = 0 → E = E_matrix."""
        E = halpin_tsai(230e9, 3.5e9, 0.0, 2)
        self.assertAlmostEqual(E, 3.5e9, places=0)

    def test_higher_fraction_gives_higher_modulus(self):
        """More filler → stiffer composite."""
        E_low = halpin_tsai(230e9, 3.5e9, 0.2, 2)
        E_high = halpin_tsai(230e9, 3.5e9, 0.5, 2)
        self.assertGreater(E_high, E_low)

    def test_higher_xi_gives_higher_modulus(self):
        """Longer fibers (higher ξ) → more reinforcement."""
        E_sphere = halpin_tsai(230e9, 3.5e9, 0.3, 2)
        E_fiber = halpin_tsai(230e9, 3.5e9, 0.3, 100)
        self.assertGreater(E_fiber, E_sphere)


class TestGibsonAshby(unittest.TestCase):
    """Foam mechanics."""

    def test_solid_gives_full_modulus(self):
        """Relative density 1.0 → E = E_solid."""
        E = gibson_ashby_modulus(200e9, 1.0)
        self.assertAlmostEqual(E, 200e9, places=0)

    def test_zero_density_gives_zero(self):
        """Relative density 0 → E = 0."""
        E = gibson_ashby_modulus(200e9, 0.0)
        self.assertEqual(E, 0.0)

    def test_low_density_foam_much_weaker(self):
        """5% relative density → E ≈ 0.25% of solid (n=2)."""
        E = gibson_ashby_modulus(200e9, 0.05, 'open')
        self.assertLess(E, 200e9 * 0.01)
        self.assertGreater(E, 0)

    def test_open_stiffer_than_closed_at_low_density(self):
        """Open-cell (n=2) stiffer than closed-cell (n=3) at low ρ_rel."""
        # At ρ_rel = 0.1: open C×0.01 = 0.01, closed 0.3×0.001 = 0.0003
        E_open = gibson_ashby_modulus(100e9, 0.1, 'open')
        E_closed = gibson_ashby_modulus(100e9, 0.1, 'closed')
        self.assertGreater(E_open, E_closed)

    def test_strength_positive(self):
        """Foam strength should be positive for any valid input."""
        s = gibson_ashby_strength(300e6, 0.1, 'open')
        self.assertGreater(s, 0)


class TestTransport(unittest.TestCase):
    """Density, CTE, thermal conductivity."""

    def test_density_exact(self):
        """Rule of mixtures for density is exact."""
        rho = density_rule_of_mixtures([2700, 1200], [0.6, 0.4])
        self.assertAlmostEqual(rho, 0.6 * 2700 + 0.4 * 1200, places=5)

    def test_thermal_expansion_weighted(self):
        """Turner CTE is modulus-weighted."""
        # If both phases have same alpha, result should be that alpha
        alpha = thermal_expansion_voigt(
            [10e-6, 10e-6], [200e9, 50e9], [0.5, 0.5]
        )
        self.assertAlmostEqual(alpha, 10e-6, places=10)

    def test_stiffer_phase_dominates_expansion(self):
        """High-modulus phase should dominate composite CTE."""
        alpha = thermal_expansion_voigt(
            [5e-6, 50e-6], [200e9, 3e9], [0.6, 0.4]
        )
        # Should be much closer to 5e-6 than to 50e-6
        self.assertLess(alpha, 15e-6)

    def test_conductivity_bounds(self):
        """Thermal conductivity bounds should be ordered."""
        k_lo, k_hi = thermal_conductivity_bounds([200, 1], [0.5, 0.5])
        self.assertLess(k_lo, k_hi)


class TestNamedComposites(unittest.TestCase):
    """Named composites from COMPOSITES database vs MEASURED values."""

    def test_cfrp_modulus_order_of_magnitude(self):
        """CFRP unidirectional: E ≈ 130-150 GPa (MEASURED)."""
        E = composite_modulus('cfrp_unidirectional') / 1e9
        self.assertGreater(E, 100)
        self.assertLess(E, 200)

    def test_cfrp_density(self):
        """CFRP: ρ ≈ 1560 kg/m³ (MEASURED)."""
        rho = composite_density('cfrp_unidirectional')
        self.assertAlmostEqual(rho, 1560, delta=100)

    def test_gfrp_modulus(self):
        """GFRP unidirectional: E ≈ 38-50 GPa (MEASURED)."""
        E = composite_modulus('gfrp_unidirectional') / 1e9
        self.assertGreater(E, 30)
        self.assertLess(E, 60)

    def test_concrete_modulus(self):
        """Concrete: E ≈ 25-45 GPa (MEASURED)."""
        E = composite_modulus('concrete') / 1e9
        self.assertGreater(E, 20)
        self.assertLess(E, 60)

    def test_wc_cobalt_very_stiff(self):
        """WC-Co: E ≈ 550-650 GPa (MEASURED)."""
        E = composite_modulus('wc_cobalt') / 1e9
        self.assertGreater(E, 400)
        self.assertLess(E, 750)

    def test_cfrp_highest_specific_stiffness(self):
        """CFRP should have highest specific stiffness (the whole point)."""
        ss_cfrp = specific_stiffness('cfrp_unidirectional')
        ss_concrete = specific_stiffness('concrete')
        ss_wc = specific_stiffness('wc_cobalt')
        self.assertGreater(ss_cfrp, ss_concrete)
        self.assertGreater(ss_cfrp, ss_wc)

    def test_chopped_weaker_than_continuous(self):
        """Chopped CFRP should be weaker than unidirectional."""
        E_cont = composite_modulus('cfrp_unidirectional')
        E_chop = composite_modulus('cfrp_chopped')
        self.assertGreater(E_cont, E_chop)

    def test_foam_much_weaker_than_solid(self):
        """Al foam at 10% density should be much weaker than solid Al."""
        E_foam = composite_modulus('al_foam') / 1e9
        self.assertLess(E_foam, 5)  # < 5 GPa (solid Al is 70 GPa)
        self.assertGreater(E_foam, 0)

    def test_all_densities_positive(self):
        """Every composite should have positive density."""
        for key in COMPOSITES:
            with self.subTest(composite=key):
                self.assertGreater(composite_density(key), 0)

    def test_all_moduli_positive(self):
        """Every composite should have positive modulus."""
        for key in COMPOSITES:
            with self.subTest(composite=key):
                self.assertGreater(composite_modulus(key), 0)


class TestCFRPExpansion(unittest.TestCase):
    """CFRP thermal expansion — carbon fiber is negative CTE."""

    def test_cfrp_low_expansion(self):
        """CFRP should have very low CTE (carbon fiber compensates epoxy)."""
        alpha = composite_expansion('cfrp_unidirectional')
        # Carbon fiber α ≈ -0.4 ppm/K, epoxy α ≈ 60 ppm/K
        # With 60% carbon at 230 GPa dominating, CTE should be near zero
        self.assertLess(abs(alpha), 10e-6)  # < 10 ppm/K


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = composite_report('cfrp_unidirectional')
        required = [
            'name', 'composite_key', 'f_fiber',
            'effective_modulus_GPa', 'density_kg_m3', 'CTE_1_K',
            'specific_stiffness_MNm_kg',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_non_foam_has_bounds(self):
        """Non-foam composites should have Voigt/Reuss in report."""
        r = composite_report('concrete')
        self.assertIn('voigt_GPa', r)
        self.assertIn('reuss_GPa', r)

    def test_full_report_all_composites(self):
        """Rule 9: covers every composite."""
        reports = full_report()
        self.assertEqual(set(reports.keys()), set(COMPOSITES.keys()))


if __name__ == '__main__':
    unittest.main()
