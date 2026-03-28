"""
Tests for viscoelasticity.py — time-dependent mechanical response.

Strategy:
  - Test relaxation time physics (Arrhenius temperature dependence)
  - Test Maxwell: stress relaxation, unbounded creep
  - Test Kelvin-Voigt: bounded creep, no instantaneous elastic response
  - Test SLS: bounded creep + stress relaxation (combines both)
  - Test dynamic properties: storage/loss moduli, loss tangent peaks
  - Test σ-field scaling
  - Test Rule 9: all materials in DIFFUSION_DATA
"""

import math
import unittest

from sigma_ground.field.interface.viscoelasticity import (
    relaxation_time,
    maxwell_stress_relaxation,
    maxwell_creep_strain,
    kelvin_voigt_creep,
    sls_creep,
    sls_stress_relaxation,
    storage_modulus,
    loss_modulus,
    loss_tangent,
    peak_damping_frequency,
    maxwell_viscosity,
    sigma_relaxation_ratio,
    viscoelastic_report,
    full_report,
)
from sigma_ground.field.interface.diffusion import DIFFUSION_DATA
from sigma_ground.field.interface.mechanical import youngs_modulus
from sigma_ground.field.constants import SIGMA_HERE


class TestRelaxationTime(unittest.TestCase):
    """Arrhenius relaxation time τ = (1/ν_D) exp(E_a / k_BT)."""

    def test_all_materials_positive(self):
        """τ > 0 for all materials at reasonable temperatures."""
        for key in DIFFUSION_DATA:
            with self.subTest(material=key):
                tau = relaxation_time(key, 1000.0)
                self.assertGreater(tau, 0)

    def test_decreases_with_temperature(self):
        """Higher T → faster relaxation → shorter τ."""
        for key in DIFFUSION_DATA:
            with self.subTest(material=key):
                tau_low = relaxation_time(key, 500.0)
                tau_high = relaxation_time(key, 1500.0)
                self.assertGreater(tau_low, tau_high)

    def test_room_temp_very_long(self):
        """At 300K, metallic relaxation time should be astronomical."""
        tau = relaxation_time('iron', 300.0)
        self.assertGreater(tau, 1e10, "Iron at 300K: τ should be >centuries")

    def test_near_melting_very_short(self):
        """Near melting point, τ should be microseconds to seconds."""
        # Iron melts at 1811K, test at 1800K
        tau = relaxation_time('iron', 1800.0)
        self.assertLess(tau, 1.0, "Iron near melting: τ should be < 1s")

    def test_zero_temperature_infinite(self):
        """T = 0 → τ = ∞ (no thermal activation)."""
        tau = relaxation_time('iron', 0)
        self.assertEqual(tau, float('inf'))


class TestMaxwellModel(unittest.TestCase):
    """Maxwell model: series spring-dashpot."""

    def test_stress_relaxation_decays(self):
        """Stress should decay over time at constant strain."""
        sigma_0 = 100e6  # 100 MPa
        T = 1500.0  # high enough for measurable relaxation
        t = relaxation_time('iron', T) * 3  # 3 time constants
        sigma_t = maxwell_stress_relaxation('iron', t, sigma_0, T)
        # After 3τ, stress should be ~5% of original
        self.assertLess(sigma_t, 0.1 * sigma_0)

    def test_stress_relaxation_at_t0(self):
        """At t=0, stress equals initial stress."""
        sigma_0 = 100e6
        sigma_t = maxwell_stress_relaxation('iron', 0, sigma_0, 1000.0)
        self.assertAlmostEqual(sigma_t, sigma_0, places=0)

    def test_creep_unbounded(self):
        """Maxwell creep grows without limit (linear in t)."""
        stress = 50e6
        T = 1500.0
        tau = relaxation_time('iron', T)
        eps_1 = maxwell_creep_strain('iron', tau, stress, T)
        eps_10 = maxwell_creep_strain('iron', 10 * tau, stress, T)
        # At t = 10τ, strain should be ~10× strain at t = τ (linear growth)
        self.assertGreater(eps_10, 5 * eps_1)


class TestKelvinVoigtModel(unittest.TestCase):
    """Kelvin-Voigt model: parallel spring-dashpot."""

    def test_creep_bounded(self):
        """KV creep asymptotes to σ/E (bounded)."""
        stress = 50e6
        T = 1500.0
        E = youngs_modulus('iron')
        eps_eq = stress / E  # equilibrium strain

        tau = relaxation_time('iron', T)
        eps_long = kelvin_voigt_creep('iron', 100 * tau, stress, T)

        # Should approach equilibrium strain
        self.assertAlmostEqual(eps_long, eps_eq, delta=eps_eq * 0.01)

    def test_creep_zero_at_t0(self):
        """KV has no instantaneous elastic response (ε(0) = 0)."""
        eps = kelvin_voigt_creep('iron', 0.0, 50e6, 1500.0)
        self.assertAlmostEqual(eps, 0.0, places=10)

    def test_creep_increases_monotonically(self):
        """Strain always increases under constant stress."""
        stress = 50e6
        T = 1500.0
        tau = relaxation_time('iron', T)
        times = [0.1 * tau, tau, 5 * tau, 20 * tau]
        strains = [kelvin_voigt_creep('iron', t, stress, T) for t in times]
        for i in range(len(strains) - 1):
            self.assertLess(strains[i], strains[i + 1])


class TestStandardLinearSolid(unittest.TestCase):
    """SLS (Zener) model: the physically correct one."""

    def test_creep_has_instantaneous_response(self):
        """SLS has immediate elastic strain at t=0."""
        stress = 50e6
        T = 1500.0
        eps_0 = sls_creep('iron', 0.0, stress, T)
        self.assertGreater(eps_0, 0, "SLS should have instantaneous elastic strain")

    def test_creep_bounded(self):
        """SLS creep is bounded (unlike Maxwell)."""
        stress = 50e6
        T = 1500.0
        tau = relaxation_time('iron', T)
        eps_long = sls_creep('iron', 1000 * tau, stress, T)
        eps_very_long = sls_creep('iron', 10000 * tau, stress, T)
        # Should converge
        self.assertAlmostEqual(eps_long, eps_very_long, delta=eps_long * 0.01)

    def test_creep_increases_from_initial(self):
        """SLS creep strain increases from initial elastic to equilibrium."""
        stress = 50e6
        T = 1500.0
        tau = relaxation_time('iron', T)
        eps_0 = sls_creep('iron', 0, stress, T)
        eps_inf = sls_creep('iron', 1000 * tau, stress, T)
        self.assertLess(eps_0, eps_inf)

    def test_stress_relaxation_decays_to_finite(self):
        """SLS stress relaxes from σ₀ = ε₀×E_U to σ_∞ = ε₀×E_R > 0."""
        strain = 0.001
        T = 1500.0
        tau = relaxation_time('iron', T)

        sigma_0 = sls_stress_relaxation('iron', 0, strain, T)
        sigma_inf = sls_stress_relaxation('iron', 1000 * tau, strain, T)

        self.assertGreater(sigma_0, sigma_inf, "Stress should relax")
        self.assertGreater(sigma_inf, 0, "SLS relaxes to finite stress, not zero")

    def test_all_materials_give_reasonable_creep(self):
        """Every material in DIFFUSION_DATA should produce finite strain."""
        for key in DIFFUSION_DATA:
            with self.subTest(material=key):
                eps = sls_creep(key, 1.0, 10e6, 1000.0)
                self.assertGreater(eps, 0)
                self.assertTrue(math.isfinite(eps))


class TestDynamicProperties(unittest.TestCase):
    """Storage/loss moduli and loss tangent."""

    def test_storage_modulus_between_bounds(self):
        """E' should be between E_R and E_U for all frequencies."""
        T = 1000.0
        E_U = youngs_modulus('iron')
        for omega in [0.001, 1.0, 1000.0, 1e10]:
            with self.subTest(omega=omega):
                Ep = storage_modulus('iron', omega, T)
                self.assertGreater(Ep, 0)
                self.assertLessEqual(Ep, E_U * 1.01)  # small tolerance

    def test_storage_increases_with_frequency(self):
        """Higher frequency → less time to relax → stiffer (more storage)."""
        T = 1000.0
        Ep_low = storage_modulus('iron', 0.001, T)
        Ep_high = storage_modulus('iron', 1e10, T)
        self.assertLess(Ep_low, Ep_high)

    def test_loss_modulus_positive(self):
        """E" ≥ 0 always (dissipation is non-negative)."""
        for key in DIFFUSION_DATA:
            with self.subTest(material=key):
                Epp = loss_modulus(key, 1.0, 1000.0)
                self.assertGreaterEqual(Epp, 0)

    def test_loss_tangent_peaks_at_resonance(self):
        """tan δ should be highest near ω = 1/τ."""
        T = 1000.0
        omega_peak = peak_damping_frequency('iron', T)
        if omega_peak > 0 and omega_peak < 1e30:
            # Compare tan δ at peak vs 100× away
            tan_peak = loss_tangent('iron', omega_peak, T)
            tan_far = loss_tangent('iron', omega_peak * 100, T)
            self.assertGreater(tan_peak, tan_far)

    def test_loss_tangent_low_at_room_temp(self):
        """Metals at room temperature: tan δ ≈ 10⁻⁴ or less at 1 Hz."""
        tan_d = loss_tangent('iron', 2 * math.pi, 300.0)
        self.assertLess(tan_d, 0.01,
            "Iron at 300K, 1 Hz: should have very low damping")


class TestMaxwellViscosity(unittest.TestCase):
    """η = E × τ."""

    def test_viscosity_positive(self):
        for key in DIFFUSION_DATA:
            with self.subTest(material=key):
                eta = maxwell_viscosity(key, 1000.0)
                self.assertGreater(eta, 0)

    def test_viscosity_decreases_with_T(self):
        """Higher T → shorter τ → lower viscosity."""
        eta_low = maxwell_viscosity('iron', 800.0)
        eta_high = maxwell_viscosity('iron', 1600.0)
        self.assertGreater(eta_low, eta_high)

    def test_room_temp_solid_like(self):
        """At 300K, η should be astronomical (solid behavior)."""
        eta = maxwell_viscosity('iron', 300.0)
        self.assertGreater(eta, 1e15, "Iron at 300K: effectively solid")


class TestSigmaFieldScaling(unittest.TestCase):
    """σ-field effects on relaxation."""

    def test_ratio_one_at_sigma_here(self):
        for key in DIFFUSION_DATA:
            with self.subTest(material=key):
                ratio = sigma_relaxation_ratio(key, 1000.0, SIGMA_HERE)
                self.assertAlmostEqual(ratio, 1.0, places=3)

    def test_nonzero_sigma_changes_tau(self):
        """At σ > 0, relaxation time should change."""
        ratio = sigma_relaxation_ratio('iron', 1000.0, 0.1)
        self.assertNotAlmostEqual(ratio, 1.0, places=2)


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = viscoelastic_report('iron', T=1000.0, omega=1.0)
        required = [
            'material', 'T_K', 'omega_rad_s', 'sigma',
            'relaxation_time_s', 'youngs_modulus_GPa', 'relaxed_modulus_GPa',
            'relaxation_ratio_ER_EU', 'storage_modulus_GPa', 'loss_modulus_GPa',
            'loss_tangent', 'peak_damping_freq_rad_s', 'maxwell_viscosity_Pa_s',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report_all_materials(self):
        """Rule 9: covers every material in DIFFUSION_DATA."""
        reports = full_report(T=1000.0)
        self.assertEqual(set(reports.keys()), set(DIFFUSION_DATA.keys()))


if __name__ == '__main__':
    unittest.main()
