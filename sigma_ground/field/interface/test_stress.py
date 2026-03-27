"""
Tests for the stress physics module — fatigue, fracture, creep.

Test structure:
  1. Griffith fracture toughness — derived from E and γ_s
  2. Stress intensity factor — K_I = σ√(πa)
  3. Critical crack length — when K_I reaches K_Ic
  4. Fatigue S-N — Basquin power law
  5. Paris law — crack growth rate
  6. Power-law creep — strain rate and rupture time
  7. Larson-Miller parameter — time-temperature equivalence
  8. σ-dependence — toughness and fatigue shift
  9. Rule 9 — every material has every field
  10. Nagatha export
"""

import math
import unittest

from .stress import (
    griffith_toughness,
    fracture_toughness,
    stress_intensity,
    critical_crack_length,
    fatigue_life,
    fatigue_strength,
    paris_crack_growth_rate,
    paris_remaining_life,
    creep_strain_rate,
    creep_rupture_time,
    larson_miller_parameter,
    sigma_fracture_toughness_shift,
    sigma_fatigue_shift,
    stress_properties,
    STRESS_DATA,
)
from .mechanical import youngs_modulus


class TestGriffithToughness(unittest.TestCase):
    """Griffith K_Ic from Young's modulus and surface energy."""

    def test_positive_for_all(self):
        """K_Ic > 0 for all materials."""
        for key in STRESS_DATA:
            K = griffith_toughness(key)
            self.assertGreater(K, 0, f"{key}: K_Ic must be positive")

    def test_silicon_brittle(self):
        """Silicon Griffith K_Ic ~ 0.5–2 MPa√m (brittle)."""
        K = griffith_toughness('silicon')
        K_MPa = K / 1e6
        self.assertGreater(K_MPa, 0.1)
        self.assertLess(K_MPa, 5.0)

    def test_metals_higher_than_silicon(self):
        """Metals have higher Griffith K_Ic than silicon."""
        K_si = griffith_toughness('silicon')
        for key in ('iron', 'copper', 'aluminum'):
            K = griffith_toughness(key)
            self.assertGreater(K, K_si,
                f"{key}: should be tougher than silicon")

    def test_measured_exceeds_griffith_for_metals(self):
        """Measured K_Ic > Griffith for ductile metals (plastic dissipation)."""
        for key, data in STRESS_DATA.items():
            if data['K_Ic_measured'] is not None:
                K_griff = griffith_toughness(key)
                K_meas = data['K_Ic_measured']
                # For ductile metals, measured >> Griffith
                # For brittle silicon, they can be comparable
                if key != 'silicon':
                    self.assertGreater(K_meas, K_griff,
                        f"{key}: measured K_Ic should exceed Griffith")


class TestStressIntensity(unittest.TestCase):
    """K_I = σ √(πa) — Mode I stress intensity factor."""

    def test_basic_calculation(self):
        """Known values: σ=100 MPa, a=1 mm → K_I ≈ 5.6 MPa√m."""
        K_I = stress_intensity(100e6, 1e-3)
        self.assertAlmostEqual(K_I / 1e6, 5.60, delta=0.1)

    def test_proportional_to_stress(self):
        """K_I doubles when stress doubles."""
        K1 = stress_intensity(100e6, 1e-3)
        K2 = stress_intensity(200e6, 1e-3)
        self.assertAlmostEqual(K2 / K1, 2.0, places=10)

    def test_proportional_to_sqrt_a(self):
        """K_I × √4 when crack length × 4."""
        K1 = stress_intensity(100e6, 1e-3)
        K2 = stress_intensity(100e6, 4e-3)
        self.assertAlmostEqual(K2 / K1, 2.0, places=10)


class TestCriticalCrackLength(unittest.TestCase):
    """Critical crack length — fracture when K_I = K_Ic."""

    def test_iron_critical_crack(self):
        """Iron at 200 MPa: critical crack ~ mm scale."""
        a_c = critical_crack_length('iron', 200e6)
        self.assertGreater(a_c, 1e-4)    # > 0.1 mm
        self.assertLess(a_c, 1.0)         # < 1 m

    def test_higher_stress_shorter_crack(self):
        """Higher stress → shorter critical crack."""
        a_low = critical_crack_length('iron', 100e6)
        a_high = critical_crack_length('iron', 400e6)
        self.assertGreater(a_low, a_high)

    def test_tougher_material_longer_crack(self):
        """Iron (K_Ic=50 MPa√m) tolerates longer cracks than silicon (0.7)."""
        a_fe = critical_crack_length('iron', 100e6)
        a_si = critical_crack_length('silicon', 100e6)
        self.assertGreater(a_fe, a_si)

    def test_roundtrip_with_K_I(self):
        """K_I at critical crack length equals K_Ic."""
        stress = 200e6
        a_c = critical_crack_length('iron', stress)
        K_I = stress_intensity(stress, a_c)
        K_Ic = fracture_toughness('iron')
        self.assertAlmostEqual(K_I, K_Ic, delta=K_Ic * 1e-10)


class TestFatigueLife(unittest.TestCase):
    """Basquin S-N curve — cycles to failure."""

    def test_high_stress_short_life(self):
        """Stress near UTS → low N_f."""
        N = fatigue_life('iron', 500e6)  # near UTS of 540 MPa
        self.assertGreater(N, 0)
        self.assertLess(N, 1e4)  # low-cycle regime

    def test_low_stress_long_life(self):
        """Stress well below UTS → high N_f."""
        N = fatigue_life('iron', 100e6)  # ~19% of UTS
        self.assertGreater(N, 1e6)  # high-cycle fatigue

    def test_higher_stress_fewer_cycles(self):
        """Doubling stress amplitude greatly reduces life."""
        N1 = fatigue_life('aluminum', 30e6)
        N2 = fatigue_life('aluminum', 60e6)
        self.assertGreater(N1, N2)

    def test_mean_stress_reduces_life(self):
        """Tensile mean stress reduces fatigue life (Morrow)."""
        N_zero = fatigue_life('iron', 200e6, sigma_mean=0.0)
        N_tensile = fatigue_life('iron', 200e6, sigma_mean=100e6)
        self.assertGreater(N_zero, N_tensile)

    def test_overstress_immediate_failure(self):
        """Stress above UTS → N_f = 0."""
        N = fatigue_life('iron', 600e6)  # > UTS
        self.assertEqual(N, 0.0)

    def test_roundtrip_with_strength(self):
        """fatigue_strength at N_f ≈ original stress_amplitude."""
        N_target = 1e6
        sigma_a = fatigue_strength('iron', N_target)
        N_back = fatigue_life('iron', sigma_a)
        self.assertAlmostEqual(N_back, N_target, delta=N_target * 0.01)

    def test_all_materials(self):
        """Fatigue life computable for all materials."""
        for key, data in STRESS_DATA.items():
            N = fatigue_life(key, data['sigma_UTS_Pa'] * 0.5)
            self.assertGreater(N, 0, f"{key}: should have finite life")


class TestParisLaw(unittest.TestCase):
    """Paris law — fatigue crack growth rate."""

    def test_positive_growth(self):
        """da/dN > 0 for ΔK > 0."""
        for key in STRESS_DATA:
            rate = paris_crack_growth_rate(key, 10e6)  # 10 MPa√m
            self.assertGreater(rate, 0, f"{key}: da/dN must be positive")

    def test_power_law_scaling(self):
        """Doubling ΔK increases da/dN by 2^m."""
        m = STRESS_DATA['iron']['m_paris']
        r1 = paris_crack_growth_rate('iron', 10e6)
        r2 = paris_crack_growth_rate('iron', 20e6)
        self.assertAlmostEqual(r2 / r1, 2.0 ** m, places=5)

    def test_iron_typical_rate(self):
        """Iron at ΔK=20 MPa√m: da/dN ~ 10⁻⁹ to 10⁻⁷ m/cycle."""
        rate = paris_crack_growth_rate('iron', 20e6)
        self.assertGreater(rate, 1e-10)
        self.assertLess(rate, 1e-5)

    def test_remaining_life_positive(self):
        """Remaining life > 0 when a_initial < a_critical."""
        a_c = critical_crack_length('iron', 200e6)
        N = paris_remaining_life('iron', a_c * 0.1, a_c, 200e6)
        self.assertGreater(N, 0)

    def test_remaining_life_zero_when_already_critical(self):
        """N = 0 if initial crack already at critical."""
        a_c = critical_crack_length('iron', 200e6)
        N = paris_remaining_life('iron', a_c, a_c, 200e6)
        self.assertEqual(N, 0.0)


class TestCreep(unittest.TestCase):
    """Power-law creep — strain rate and rupture time."""

    def test_positive_rate(self):
        """Creep rate > 0 at finite stress and temperature."""
        for key in STRESS_DATA:
            rate = creep_strain_rate(key, 100e6, 800)
            self.assertGreaterEqual(rate, 0, f"{key}: rate must be >= 0")

    def test_zero_at_zero_temperature(self):
        """No creep at T=0."""
        rate = creep_strain_rate('iron', 100e6, 0)
        self.assertEqual(rate, 0.0)

    def test_increases_with_temperature(self):
        """Creep rate increases with temperature (thermally activated)."""
        r_low = creep_strain_rate('iron', 100e6, 600)
        r_high = creep_strain_rate('iron', 100e6, 900)
        self.assertGreater(r_high, r_low)

    def test_increases_with_stress(self):
        """Creep rate increases with stress."""
        r_low = creep_strain_rate('iron', 50e6, 800)
        r_high = creep_strain_rate('iron', 200e6, 800)
        self.assertGreater(r_high, r_low)

    def test_stress_exponent(self):
        """Doubling stress increases rate by 2^n."""
        n = STRESS_DATA['copper']['n_creep']
        r1 = creep_strain_rate('copper', 100e6, 800)
        r2 = creep_strain_rate('copper', 200e6, 800)
        if r1 > 0:
            self.assertAlmostEqual(r2 / r1, 2.0 ** n, places=3)

    def test_rupture_time_inverse_of_rate(self):
        """Rupture time ≈ ε_limit / ε̇."""
        rate = creep_strain_rate('iron', 100e6, 800)
        t_r = creep_rupture_time('iron', 100e6, 800, strain_limit=0.01)
        if rate > 0:
            self.assertAlmostEqual(t_r, 0.01 / rate, places=5)

    def test_tungsten_slow_creep(self):
        """Tungsten (high Q) creeps much slower than aluminum (low Q)."""
        r_W = creep_strain_rate('tungsten', 100e6, 800)
        r_Al = creep_strain_rate('aluminum', 100e6, 800)
        self.assertLess(r_W, r_Al)

    def test_silicon_extremely_slow(self):
        """Silicon (covalent, Q=5 eV) has negligible creep at 800 K."""
        rate = creep_strain_rate('silicon', 100e6, 800)
        self.assertLess(rate, 1e-20)


class TestLarsonMiller(unittest.TestCase):
    """Larson-Miller parameter — time-temperature equivalence."""

    def test_basic_value(self):
        """P = T × (20 + log10(t_r)), known inputs."""
        P = larson_miller_parameter(800, 1000)  # 800K, 1000 hours
        # P = 800 × (20 + 3) = 18400
        self.assertAlmostEqual(P, 18400, delta=1)

    def test_higher_T_higher_P(self):
        """Higher temperature → higher P."""
        P1 = larson_miller_parameter(700, 1000)
        P2 = larson_miller_parameter(900, 1000)
        self.assertGreater(P2, P1)

    def test_longer_time_higher_P(self):
        """Longer time → higher P."""
        P1 = larson_miller_parameter(800, 100)
        P2 = larson_miller_parameter(800, 10000)
        self.assertGreater(P2, P1)


class TestSigmaEffects(unittest.TestCase):
    """σ-field shifts toughness and fatigue through moduli."""

    def test_toughness_unchanged_at_zero(self):
        """K_Ic(σ=0) = K_Ic(0)."""
        K_0 = fracture_toughness('iron', 0.0)
        K_s = sigma_fracture_toughness_shift('iron', 0.0)
        self.assertAlmostEqual(K_0, K_s, places=10)

    def test_toughness_shifts_with_sigma(self):
        """K_Ic(σ>0) ≠ K_Ic(0) — modulus shift."""
        K_0 = fracture_toughness('iron', 0.0)
        K_s = sigma_fracture_toughness_shift('iron', 0.1)
        self.assertNotAlmostEqual(K_0, K_s, places=3)

    def test_fatigue_shifts_with_sigma(self):
        """σ_UTS(σ>0) ≠ σ_UTS(0) — modulus shift."""
        uts_0 = STRESS_DATA['iron']['sigma_UTS_Pa']
        uts_s = sigma_fatigue_shift('iron', 0.1)
        self.assertNotAlmostEqual(uts_0, uts_s, places=0)

    def test_creep_shifts_with_sigma(self):
        """Creep rate changes with σ-field (Q shift)."""
        r_0 = creep_strain_rate('iron', 100e6, 800, sigma=0.0)
        r_s = creep_strain_rate('iron', 100e6, 800, sigma=0.1)
        self.assertNotAlmostEqual(r_0, r_s, places=10)


class TestRule9Stress(unittest.TestCase):
    """Rule 9 — every material has every field."""

    _REQUIRED_FIELDS = {
        'sigma_UTS_Pa', 'b_fatigue', 'C_paris', 'm_paris',
        'K_Ic_measured', 'Q_creep_eV', 'n_creep', 'A_creep',
    }

    def test_all_materials_present(self):
        """Every material in MECHANICAL_DATA has STRESS_DATA."""
        from .mechanical import MECHANICAL_DATA
        for key in MECHANICAL_DATA:
            self.assertIn(key, STRESS_DATA,
                f"{key}: in MECHANICAL_DATA but missing from STRESS_DATA")

    def test_all_fields_present(self):
        """Every entry has every required field."""
        for key, data in STRESS_DATA.items():
            for field in self._REQUIRED_FIELDS:
                self.assertIn(field, data,
                    f"{key}: missing field '{field}'")

    def test_positive_physical_values(self):
        """Physical quantities are positive (except b_fatigue which is negative)."""
        for key, data in STRESS_DATA.items():
            self.assertGreater(data['sigma_UTS_Pa'], 0, f"{key}: UTS")
            self.assertLess(data['b_fatigue'], 0, f"{key}: b must be < 0")
            self.assertGreater(data['C_paris'], 0, f"{key}: C_paris")
            self.assertGreater(data['m_paris'], 0, f"{key}: m_paris")
            self.assertGreater(data['Q_creep_eV'], 0, f"{key}: Q_creep")
            self.assertGreater(data['n_creep'], 0, f"{key}: n_creep")
            self.assertGreater(data['A_creep'], 0, f"{key}: A_creep")


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_basic_export(self):
        """Export includes required fields."""
        props = stress_properties('iron')
        self.assertIn('sigma_UTS_Pa', props)
        self.assertIn('K_Ic_Pa_sqrtm', props)
        self.assertIn('K_Ic_source', props)
        self.assertIn('origin_tag', props)

    def test_with_crack(self):
        """Export with stress and crack includes fracture assessment."""
        props = stress_properties('iron', applied_stress=200e6,
                                  crack_length=1e-3)
        self.assertIn('K_I_Pa_sqrtm', props)
        self.assertIn('critical_crack_m', props)
        self.assertIn('will_fracture', props)

    def test_with_temperature(self):
        """Export with temperature includes creep data."""
        props = stress_properties('iron', applied_stress=100e6,
                                  temperature=800)
        self.assertIn('creep_rate_1_s', props)
        self.assertIn('creep_rupture_s', props)

    def test_all_materials_export(self):
        """All materials produce valid export."""
        for key in STRESS_DATA:
            props = stress_properties(key)
            self.assertIn('K_Ic_Pa_sqrtm', props)
            self.assertGreater(props['K_Ic_Pa_sqrtm'], 0)


if __name__ == '__main__':
    unittest.main()
