"""
Tests for the electronics module — transport, semiconductors, junctions.

Test structure:
  1. Bloch-Grüneisen resistivity — ρ(T) shape, calibration, limits
  2. Carrier mobility — Drude model
  3. Hall effect — coefficient and voltage
  4. Intrinsic carrier concentration — n_i(T) for Si, Ge, GaAs
  5. Doped semiconductors — carrier types, Fermi level
  6. p-n junction — V_bi, depletion width, diode I-V
  7. Geometric capacitance — parallel plate, coaxial, spherical
  8. σ-dependence — resistivity shift through Θ_D
  9. Rule 9 — every material has every field
  10. Nagatha export
"""

import math
import unittest

from .electronics import (
    resistivity,
    resistivity_sigma,
    carrier_mobility,
    mean_free_time,
    mean_free_path,
    free_electron_density,
    hall_coefficient,
    hall_voltage,
    band_gap,
    intrinsic_carrier_concentration,
    effective_dos_conduction,
    effective_dos_valence,
    carrier_concentration,
    fermi_level_from_intrinsic,
    built_in_voltage,
    depletion_width,
    junction_capacitance,
    diode_saturation_current,
    diode_current,
    parallel_plate_capacitance,
    coaxial_capacitance,
    spherical_capacitance,
    energy_stored,
    metal_transport_properties,
    semiconductor_properties,
    METAL_TRANSPORT,
    SEMICONDUCTORS,
)
from ..constants import E_CHARGE, K_B


class TestBlochGruneisen(unittest.TestCase):
    """Bloch-Grüneisen resistivity ρ(T)."""

    def test_reproduces_300K(self):
        """ρ(300K) matches stored measured value (calibration check)."""
        for key, data in METAL_TRANSPORT.items():
            if data['rho_300'] > 1.0:
                continue  # Skip semiconductors
            rho = resistivity(key, 300.0)
            self.assertAlmostEqual(rho / data['rho_300'], 1.0, places=3,
                msg=f"{key}: ρ(300)={rho:.3e}, expected={data['rho_300']:.3e}")

    def test_residual_at_zero(self):
        """ρ(0) = ρ₀ (residual resistivity)."""
        for key, data in METAL_TRANSPORT.items():
            if data['rho_300'] > 1.0:
                continue
            rho = resistivity(key, 0.0)
            self.assertAlmostEqual(rho, data['rho_0'], places=15,
                msg=f"{key}: ρ(0)={rho:.3e}")

    def test_increases_with_temperature(self):
        """ρ(T) increases monotonically with T (phonon scattering)."""
        for key, data in METAL_TRANSPORT.items():
            if data['rho_300'] > 1.0:
                continue
            prev = resistivity(key, 1.0)
            for T in [10, 50, 100, 200, 300, 500]:
                rho = resistivity(key, T)
                self.assertGreaterEqual(rho, prev - 1e-20,
                    f"{key}: ρ decreased at T={T}")
                prev = rho

    def test_linear_at_high_T(self):
        """Above Θ_D: ρ(T) ∝ T (phonon population ∝ T)."""
        # Copper: Θ_D=343K, check linearity above 500K
        rho_500 = resistivity('copper', 500)
        rho_1000 = resistivity('copper', 1000)
        ratio = rho_1000 / rho_500
        # Should be close to 2.0 (linear regime)
        self.assertGreater(ratio, 1.7)
        self.assertLess(ratio, 2.3)

    def test_T5_at_low_T(self):
        """Below ~Θ_D/10: ρ − ρ₀ ∝ T⁵ (phonon freeze-out)."""
        # Copper: Θ_D=343, test at T=10K and T=20K
        rho_0 = METAL_TRANSPORT['copper']['rho_0']
        delta_10 = resistivity('copper', 10) - rho_0
        delta_20 = resistivity('copper', 20) - rho_0
        if delta_10 > 0:
            ratio = delta_20 / delta_10
            # Should be close to 2^5 = 32
            self.assertGreater(ratio, 20)
            self.assertLess(ratio, 40)

    def test_copper_vs_aluminum(self):
        """Copper has lower resistivity than aluminum above residual regime."""
        # Below ~50K both metals are dominated by residual ρ₀ which depends
        # on sample purity, not intrinsic physics. Compare at higher T.
        for T in [100, 300, 500]:
            rho_Cu = resistivity('copper', T)
            rho_Al = resistivity('aluminum', T)
            self.assertLess(rho_Cu, rho_Al,
                f"T={T}: Cu should be more conductive than Al")

    def test_all_metals(self):
        """ρ(T) computable for all metals."""
        for key in METAL_TRANSPORT:
            rho = resistivity(key, 300)
            self.assertGreater(rho, 0, f"{key}: ρ must be positive")


class TestCarrierMobility(unittest.TestCase):
    """Drude mobility μ = 1/(n_e e ρ)."""

    def test_copper_300K(self):
        """Copper mobility at 300K ~ 30-50 cm²/(V·s)."""
        mu = carrier_mobility('copper', 300)
        mu_cm2 = mu * 1e4  # m²/(V·s) → cm²/(V·s)
        self.assertGreater(mu_cm2, 10)
        self.assertLess(mu_cm2, 100)

    def test_increases_on_cooling(self):
        """Mobility increases as T decreases (less scattering)."""
        mu_300 = carrier_mobility('copper', 300)
        mu_100 = carrier_mobility('copper', 100)
        self.assertGreater(mu_100, mu_300)

    def test_mean_free_path_reasonable(self):
        """Mean free path at 300K ~ 10-100 nm for metals."""
        for key in ('copper', 'aluminum', 'gold'):
            mfp = mean_free_path(key, 300)
            self.assertGreater(mfp, 1e-9, f"{key}: ℓ too short")
            self.assertLess(mfp, 1e-6, f"{key}: ℓ too long")

    def test_mean_free_time_positive(self):
        """τ > 0 for all metals."""
        for key, data in METAL_TRANSPORT.items():
            if data['rho_300'] > 1.0:
                continue
            tau = mean_free_time(key, 300)
            self.assertGreater(tau, 0, f"{key}: τ must be positive")


class TestHallEffect(unittest.TestCase):
    """Hall coefficient and voltage."""

    def test_negative_for_electrons(self):
        """R_H < 0 for electron carriers."""
        for key in METAL_TRANSPORT:
            R_H = hall_coefficient(key)
            self.assertLess(R_H, 0, f"{key}: R_H should be negative")

    def test_copper_R_H(self):
        """Copper R_H ≈ −7.3×10⁻¹¹ m³/C (1 electron/atom)."""
        R_H = hall_coefficient('copper')
        self.assertAlmostEqual(R_H * 1e11, -7.3, delta=1.0)

    def test_hall_voltage_sign(self):
        """Hall voltage negative for electron carriers (positive I, B)."""
        V_H = hall_voltage('copper', current=1.0, B_field=1.0,
                           thickness=1e-3)
        self.assertLess(V_H, 0)

    def test_hall_voltage_proportional_to_B(self):
        """V_H ∝ B."""
        V1 = hall_voltage('copper', 1.0, 1.0, 1e-3)
        V2 = hall_voltage('copper', 1.0, 2.0, 1e-3)
        self.assertAlmostEqual(V2 / V1, 2.0, places=10)


class TestBandGap(unittest.TestCase):
    """Temperature-dependent band gap (Varshni)."""

    def test_silicon_300K(self):
        """Si band gap at 300K ≈ 1.12 eV."""
        E_g = band_gap('silicon', 300)
        self.assertAlmostEqual(E_g, 1.12, delta=0.02)

    def test_germanium_300K(self):
        """Ge band gap at 300K ≈ 0.66 eV."""
        E_g = band_gap('germanium', 300)
        self.assertAlmostEqual(E_g, 0.66, delta=0.02)

    def test_GaAs_300K(self):
        """GaAs band gap at 300K ≈ 1.42 eV."""
        E_g = band_gap('gallium_arsenide', 300)
        self.assertAlmostEqual(E_g, 1.42, delta=0.03)

    def test_gap_decreases_with_T(self):
        """Band gap shrinks with temperature."""
        for key in SEMICONDUCTORS:
            E_0 = band_gap(key, 0)
            E_300 = band_gap(key, 300)
            self.assertGreater(E_0, E_300,
                f"{key}: gap should shrink with T")

    def test_all_positive(self):
        """Band gap positive for all semiconductors at 300K."""
        for key in SEMICONDUCTORS:
            self.assertGreater(band_gap(key, 300), 0, key)


class TestIntrinsicCarrier(unittest.TestCase):
    """Intrinsic carrier concentration n_i(T)."""

    def test_silicon_300K(self):
        """Si n_i at 300K ≈ 1.0–1.5 × 10¹⁶ m⁻³."""
        n_i = intrinsic_carrier_concentration('silicon', 300)
        self.assertGreater(n_i, 5e15)
        self.assertLess(n_i, 3e16)

    def test_germanium_higher_than_silicon(self):
        """Ge n_i > Si n_i at 300K (smaller gap)."""
        n_si = intrinsic_carrier_concentration('silicon', 300)
        n_ge = intrinsic_carrier_concentration('germanium', 300)
        self.assertGreater(n_ge, n_si)

    def test_GaN_much_lower(self):
        """GaN n_i << Si n_i at 300K (wide gap)."""
        n_si = intrinsic_carrier_concentration('silicon', 300)
        n_GaN = intrinsic_carrier_concentration('gallium_nitride', 300)
        self.assertLess(n_GaN, n_si * 1e-10)

    def test_increases_with_temperature(self):
        """n_i increases exponentially with T."""
        n_300 = intrinsic_carrier_concentration('silicon', 300)
        n_400 = intrinsic_carrier_concentration('silicon', 400)
        self.assertGreater(n_400, n_300 * 10)  # Exponential increase

    def test_zero_at_zero_T(self):
        """n_i = 0 at T = 0 (all carriers frozen out)."""
        n_i = intrinsic_carrier_concentration('silicon', 0)
        self.assertEqual(n_i, 0.0)

    def test_all_semiconductors(self):
        """n_i computable for all semiconductors."""
        for key in SEMICONDUCTORS:
            n_i = intrinsic_carrier_concentration(key, 300)
            self.assertGreater(n_i, 0, f"{key}: n_i must be positive at 300K")


class TestDoping(unittest.TestCase):
    """Doped semiconductor carrier concentration."""

    def test_n_type(self):
        """N_D doping → n ≈ N_D, p << n."""
        N_D = 1e22  # 10¹⁶ cm⁻³ in m⁻³
        n, p = carrier_concentration('silicon', 300, N_D=N_D)
        self.assertAlmostEqual(n / N_D, 1.0, delta=0.01)
        self.assertLess(p, n * 1e-6)

    def test_p_type(self):
        """N_A doping → p ≈ N_A, n << p."""
        N_A = 1e22
        n, p = carrier_concentration('silicon', 300, N_A=N_A)
        self.assertAlmostEqual(p / N_A, 1.0, delta=0.01)
        self.assertLess(n, p * 1e-6)

    def test_intrinsic(self):
        """No doping → n = p = n_i."""
        n, p = carrier_concentration('silicon', 300)
        n_i = intrinsic_carrier_concentration('silicon', 300)
        self.assertAlmostEqual(n / n_i, 1.0, delta=0.01)
        self.assertAlmostEqual(p / n_i, 1.0, delta=0.01)

    def test_mass_action(self):
        """n × p = n_i² always holds."""
        n_i = intrinsic_carrier_concentration('silicon', 300)
        n, p = carrier_concentration('silicon', 300, N_D=1e22)
        self.assertAlmostEqual(n * p / n_i ** 2, 1.0, delta=0.02)

    def test_fermi_level_n_type(self):
        """E_F above E_Fi for n-type."""
        dE = fermi_level_from_intrinsic('silicon', 300, N_D=1e22)
        self.assertGreater(dE, 0)

    def test_fermi_level_p_type(self):
        """E_F below E_Fi for p-type."""
        dE = fermi_level_from_intrinsic('silicon', 300, N_A=1e22)
        self.assertLess(dE, 0)


class TestPNJunction(unittest.TestCase):
    """p-n junction physics."""

    _ND = 1e22   # 10¹⁶ cm⁻³
    _NA = 1e22

    def test_built_in_voltage_silicon(self):
        """Si V_bi ~ 0.6–0.9 V for moderate doping."""
        V_bi = built_in_voltage('silicon', self._ND, self._NA)
        self.assertGreater(V_bi, 0.5)
        self.assertLess(V_bi, 1.0)

    def test_V_bi_increases_with_doping(self):
        """Higher doping → higher V_bi."""
        V_low = built_in_voltage('silicon', 1e21, 1e21)
        V_high = built_in_voltage('silicon', 1e23, 1e23)
        self.assertGreater(V_high, V_low)

    def test_depletion_width_positive(self):
        """W > 0 at zero bias."""
        W = depletion_width('silicon', self._ND, self._NA)
        self.assertGreater(W, 0)

    def test_depletion_width_reverse_bias(self):
        """Reverse bias widens depletion region."""
        W_0 = depletion_width('silicon', self._ND, self._NA, 0.0)
        W_rev = depletion_width('silicon', self._ND, self._NA, -5.0)
        self.assertGreater(W_rev, W_0)

    def test_depletion_width_forward_shrinks(self):
        """Forward bias narrows depletion region."""
        W_0 = depletion_width('silicon', self._ND, self._NA, 0.0)
        W_fwd = depletion_width('silicon', self._ND, self._NA, 0.3)
        self.assertLess(W_fwd, W_0)

    def test_junction_capacitance_positive(self):
        """C_j > 0."""
        C = junction_capacitance('silicon', self._ND, self._NA, 1e-6)
        self.assertGreater(C, 0)

    def test_capacitance_increases_reverse(self):
        """C_j decreases with reverse bias (wider W)."""
        C_0 = junction_capacitance('silicon', self._ND, self._NA, 1e-6, 0.0)
        C_rev = junction_capacitance('silicon', self._ND, self._NA, 1e-6, -5.0)
        self.assertLess(C_rev, C_0)

    def test_diode_forward_exponential(self):
        """Forward bias: I grows exponentially."""
        I_0 = 1e-12  # 1 pA
        I_300 = diode_current(I_0, 0.3)
        I_600 = diode_current(I_0, 0.6)
        self.assertGreater(I_600, I_300 * 1000)  # ~exp(11.6) ratio

    def test_diode_reverse_saturation(self):
        """Reverse bias: I → −I₀."""
        I_0 = 1e-12
        I_rev = diode_current(I_0, -1.0)
        self.assertAlmostEqual(I_rev, -I_0, delta=I_0 * 0.01)

    def test_diode_zero_at_zero(self):
        """I(V=0) = 0."""
        I_0 = 1e-12
        self.assertAlmostEqual(diode_current(I_0, 0.0), 0.0, places=20)

    def test_saturation_current_positive(self):
        """I₀ > 0."""
        I_0 = diode_saturation_current('silicon', self._ND, self._NA, 1e-6)
        self.assertGreater(I_0, 0)


class TestGeometricCapacitance(unittest.TestCase):
    """Geometric capacitors from Gauss's law."""

    def test_parallel_plate_basic(self):
        """1 m² plates, 1 mm apart, air: C ≈ 8.85 nF."""
        C = parallel_plate_capacitance(1.0, 1e-3)
        self.assertAlmostEqual(C * 1e9, 8.85, delta=0.1)

    def test_dielectric_multiplier(self):
        """ε_r doubles C."""
        C1 = parallel_plate_capacitance(1.0, 1e-3, 1.0)
        C2 = parallel_plate_capacitance(1.0, 1e-3, 2.0)
        self.assertAlmostEqual(C2 / C1, 2.0, places=10)

    def test_coaxial_positive(self):
        """Coaxial capacitance > 0."""
        C = coaxial_capacitance(1.0, 1e-3, 5e-3)
        self.assertGreater(C, 0)

    def test_spherical_positive(self):
        """Spherical capacitance > 0."""
        C = spherical_capacitance(0.01, 0.02)
        self.assertGreater(C, 0)

    def test_energy_stored(self):
        """U = ½CV²."""
        C = 1e-6  # 1 μF
        V = 10.0
        U = energy_stored(C, V)
        self.assertAlmostEqual(U, 0.5e-6 * 100, places=10)


class TestSigmaEffects(unittest.TestCase):
    """σ-field shifts ρ(T) through Θ_D."""

    def test_identity_at_zero(self):
        """ρ(σ=0) = ρ(0)."""
        rho_0 = resistivity('copper', 300)
        rho_s = resistivity_sigma('copper', 300, 0.0)
        self.assertAlmostEqual(rho_0, rho_s, places=15)

    def test_sigma_shifts_resistivity(self):
        """Positive σ changes ρ(T) through Θ_D shift."""
        rho_0 = resistivity('copper', 300)
        rho_s = resistivity_sigma('copper', 300, 0.1)
        # Both should reproduce 300K (calibration), but at other T they differ
        # Test at non-calibration temperature
        rho_100_0 = resistivity('copper', 100)
        rho_100_s = resistivity_sigma('copper', 100, 0.1)
        # σ=0.1 shifts Θ_D by ~0.5%, so ρ differs at ~% level
        rel_diff = abs(rho_100_s - rho_100_0) / rho_100_0
        self.assertGreater(rel_diff, 0.001,
            f"σ should shift ρ(100K) by >0.1%, got {rel_diff:.4%}")


class TestRule9(unittest.TestCase):
    """Rule 9 — every entry has every field."""

    _METAL_FIELDS = {'rho_300', 'rho_0', 'theta_D', 'Z_val',
                     'rho_kg_m3', 'M_g'}
    _SC_FIELDS = {'E_g_eV', 'm_e_eff', 'm_h_eff', 'epsilon_r',
                  'E_donor_eV', 'E_acceptor_eV', 'varshni_alpha',
                  'varshni_beta'}

    def test_all_metals_complete(self):
        """Every metal has every required field."""
        for key, data in METAL_TRANSPORT.items():
            for field in self._METAL_FIELDS:
                self.assertIn(field, data, f"{key}: missing '{field}'")

    def test_all_semiconductors_complete(self):
        """Every semiconductor has every required field."""
        for key, data in SEMICONDUCTORS.items():
            for field in self._SC_FIELDS:
                self.assertIn(field, data, f"{key}: missing '{field}'")

    def test_metal_count(self):
        """At least 7 metals (matching mechanical.py)."""
        self.assertGreaterEqual(len(METAL_TRANSPORT), 7)

    def test_semiconductor_count(self):
        """At least 5 semiconductors."""
        self.assertGreaterEqual(len(SEMICONDUCTORS), 5)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_metal_export(self):
        """Metal transport export includes key fields."""
        props = metal_transport_properties('copper', 300)
        self.assertIn('resistivity_ohm_m', props)
        self.assertIn('mobility_m2_V_s', props)
        self.assertIn('hall_coefficient_m3_C', props)
        self.assertIn('origin_tag', props)

    def test_semiconductor_export(self):
        """Semiconductor export includes key fields."""
        props = semiconductor_properties('silicon', 300)
        self.assertIn('band_gap_eV', props)
        self.assertIn('n_i_m3', props)
        self.assertIn('carrier_type', props)

    def test_doped_export_has_junction(self):
        """Doped semiconductor export includes junction data."""
        props = semiconductor_properties('silicon', 300,
                                         N_D=1e22, N_A=1e22)
        self.assertIn('V_bi_V', props)
        self.assertIn('depletion_width_m', props)

    def test_all_metals_export(self):
        """All metals produce valid export."""
        for key in METAL_TRANSPORT:
            props = metal_transport_properties(key, 300)
            self.assertGreater(props['resistivity_ohm_m'], 0)

    def test_all_semiconductors_export(self):
        """All semiconductors produce valid export."""
        for key in SEMICONDUCTORS:
            props = semiconductor_properties(key, 300)
            self.assertGreater(props['band_gap_eV'], 0)


if __name__ == '__main__':
    unittest.main()
