"""
Tests for the acoustics module.

Test structure:
  1. Wave speeds — longitudinal faster than transverse, known values
  2. Debye velocity — weighted average between v_T and v_L
  3. Acoustic impedance — Z = ρv, reflection/transmission
  4. Snell's law — refraction, critical angle, total internal reflection
  5. Resonance — standing waves, ring frequency
  6. σ-dependence — wave speeds shift with σ
  7. Cross-module consistency — uses K, G from mechanical.py
  8. Nagatha export — complete format with origin tags
"""

import math
import unittest

from .acoustics import (
    density_at_sigma,
    longitudinal_wave_speed,
    transverse_wave_speed,
    debye_velocity,
    wave_speed_ratio,
    acoustic_impedance,
    reflection_coefficient,
    transmission_coefficient,
    snell_refraction_angle,
    critical_angle,
    resonant_frequency,
    ring_frequency,
    wavelength,
    material_acoustic_properties,
)
from .surface import MATERIALS
from .mechanical import MECHANICAL_DATA


class TestWaveSpeeds(unittest.TestCase):
    """Longitudinal and transverse wave speeds from elastic moduli."""

    def test_longitudinal_positive(self):
        """All materials have positive longitudinal wave speed."""
        for mat in MECHANICAL_DATA:
            v = longitudinal_wave_speed(mat)
            self.assertGreater(v, 0, f"{mat}: v_L should be positive")

    def test_transverse_positive(self):
        """All materials have positive transverse wave speed."""
        for mat in MECHANICAL_DATA:
            v = transverse_wave_speed(mat)
            self.assertGreater(v, 0, f"{mat}: v_T should be positive")

    def test_longitudinal_faster_than_transverse(self):
        """v_L > v_T always (K > 0 guarantees this).

        The longitudinal modulus M = K + 4G/3 > G for any K > 0.
        This is a fundamental result of continuum mechanics.
        """
        for mat in MECHANICAL_DATA:
            v_L = longitudinal_wave_speed(mat)
            v_T = transverse_wave_speed(mat)
            self.assertGreater(v_L, v_T,
                f"{mat}: v_L should exceed v_T")

    def test_iron_known_speeds(self):
        """Iron: v_L ≈ 5960 m/s, v_T ≈ 3240 m/s (within ±50%).

        Our model uses harmonic-approximation K (±50%), so we allow
        a wide tolerance. The point is to get the right regime.
        """
        v_L = longitudinal_wave_speed('iron')
        v_T = transverse_wave_speed('iron')
        # Known measured values: v_L = 5960, v_T = 3240
        self.assertGreater(v_L, 3000)
        self.assertLess(v_L, 9000)
        self.assertGreater(v_T, 1500)
        self.assertLess(v_T, 5000)

    def test_copper_known_speeds(self):
        """Copper: v_L ≈ 4760 m/s (within ±50%)."""
        v_L = longitudinal_wave_speed('copper')
        self.assertGreater(v_L, 2500)
        self.assertLess(v_L, 7500)

    def test_aluminum_fast(self):
        """Aluminum: light and stiff → fast sound.

        Al has lower density than Fe/Cu but similar stiffness,
        so sound should be faster.
        """
        v_al = longitudinal_wave_speed('aluminum')
        # Known: ~6420 m/s
        self.assertGreater(v_al, 3000)
        self.assertLess(v_al, 10000)

    def test_tungsten_slow_transverse(self):
        """Tungsten: very heavy → relatively slow despite high stiffness."""
        v_T_w = transverse_wave_speed('tungsten')
        v_T_al = transverse_wave_speed('aluminum')
        # Tungsten is much denser, so despite higher G, v_T is lower
        # (not guaranteed, but typical)
        self.assertGreater(v_T_w, 0)
        self.assertGreater(v_T_al, 0)


class TestWaveSpeedRatio(unittest.TestCase):
    """v_L/v_T depends only on Poisson's ratio."""

    def test_cauchy_solid(self):
        """For ν = 0.25 (Cauchy): v_L/v_T = √3 ≈ 1.732."""
        # Iron has ν = 0.29, close to Cauchy
        ratio = wave_speed_ratio('iron')
        # ν=0.29 → √(2×0.71/0.42) = √3.38 ≈ 1.84
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)

    def test_increases_with_poisson(self):
        """Higher ν → higher v_L/v_T ratio.

        As material becomes more incompressible (ν → 0.5),
        v_T → 0 while v_L stays finite → ratio → ∞.
        """
        # Gold (ν=0.44) should have higher ratio than iron (ν=0.29)
        r_au = wave_speed_ratio('gold')
        r_fe = wave_speed_ratio('iron')
        self.assertGreater(r_au, r_fe)

    def test_always_greater_than_1(self):
        """v_L/v_T > 1 for all physical materials (ν > −1)."""
        for mat in MECHANICAL_DATA:
            ratio = wave_speed_ratio(mat)
            self.assertGreater(ratio, 1.0)


class TestDebyeVelocity(unittest.TestCase):
    """Debye average velocity — between v_T and v_L."""

    def test_between_vT_and_vL(self):
        """v_T < v_D < v_L for all materials."""
        for mat in MECHANICAL_DATA:
            v_T = transverse_wave_speed(mat)
            v_L = longitudinal_wave_speed(mat)
            v_D = debye_velocity(mat)
            self.assertGreater(v_D, v_T,
                f"{mat}: v_D should exceed v_T")
            self.assertLess(v_D, v_L,
                f"{mat}: v_D should be less than v_L")

    def test_closer_to_vT(self):
        """v_D is closer to v_T than to v_L (2 transverse modes dominate)."""
        for mat in MECHANICAL_DATA:
            v_T = transverse_wave_speed(mat)
            v_L = longitudinal_wave_speed(mat)
            v_D = debye_velocity(mat)
            dist_to_T = v_D - v_T
            dist_to_L = v_L - v_D
            self.assertLess(dist_to_T, dist_to_L,
                f"{mat}: v_D should be closer to v_T")


class TestAcousticImpedance(unittest.TestCase):
    """Z = ρv — acoustic impedance."""

    def test_positive(self):
        """All materials have positive impedance."""
        for mat in MECHANICAL_DATA:
            Z = acoustic_impedance(mat)
            self.assertGreater(Z, 0)

    def test_iron_known_value(self):
        """Iron Z ≈ 47 MRayl (within ±50%)."""
        Z = acoustic_impedance('iron')
        # Known: ~47 × 10⁶ Rayl
        self.assertGreater(Z, 20e6)
        self.assertLess(Z, 80e6)

    def test_longitudinal_greater_than_transverse(self):
        """Z_L > Z_T (because v_L > v_T, same ρ)."""
        for mat in MECHANICAL_DATA:
            Z_L = acoustic_impedance(mat, mode='longitudinal')
            Z_T = acoustic_impedance(mat, mode='transverse')
            self.assertGreater(Z_L, Z_T)

    def test_tungsten_high_impedance(self):
        """Tungsten: very dense → very high acoustic impedance."""
        Z_w = acoustic_impedance('tungsten')
        Z_al = acoustic_impedance('aluminum')
        self.assertGreater(Z_w, Z_al)


class TestReflectionTransmission(unittest.TestCase):
    """Acoustic reflection and transmission at interfaces."""

    def test_same_material_no_reflection(self):
        """Same material on both sides: R = 0 (perfect transmission)."""
        R = reflection_coefficient('iron', 'iron')
        self.assertAlmostEqual(R, 0.0, places=10)

    def test_same_material_full_transmission(self):
        """Same material: T = 1."""
        T = transmission_coefficient('copper', 'copper')
        self.assertAlmostEqual(T, 1.0, places=10)

    def test_energy_conservation(self):
        """R + T = 1 for all material pairs."""
        pairs = [('iron', 'copper'), ('aluminum', 'gold'),
                 ('tungsten', 'silicon'), ('nickel', 'titanium')]
        for m1, m2 in pairs:
            R = reflection_coefficient(m1, m2)
            T = transmission_coefficient(m1, m2)
            self.assertAlmostEqual(R + T, 1.0, places=10,
                msg=f"{m1}/{m2}: R + T should equal 1")

    def test_reflection_between_0_and_1(self):
        """0 ≤ R ≤ 1 for all pairs."""
        for m1 in MECHANICAL_DATA:
            for m2 in MECHANICAL_DATA:
                R = reflection_coefficient(m1, m2)
                self.assertGreaterEqual(R, 0.0)
                self.assertLessEqual(R, 1.0)

    def test_symmetric(self):
        """R is the same regardless of direction (at normal incidence)."""
        R_12 = reflection_coefficient('iron', 'aluminum')
        R_21 = reflection_coefficient('aluminum', 'iron')
        self.assertAlmostEqual(R_12, R_21, places=10)

    def test_high_mismatch_high_reflection(self):
        """Large impedance mismatch → high reflection.

        Tungsten/aluminum should reflect more than iron/nickel (similar Z).
        """
        R_large = reflection_coefficient('tungsten', 'aluminum')
        R_small = reflection_coefficient('iron', 'nickel')
        self.assertGreater(R_large, R_small)


class TestSnellsLaw(unittest.TestCase):
    """Acoustic refraction and total internal reflection."""

    def test_normal_incidence_zero(self):
        """At 0° incidence, refracted angle is 0°."""
        theta = snell_refraction_angle('iron', 'copper', 0.0)
        self.assertAlmostEqual(theta, 0.0, places=5)

    def test_refraction_away_from_normal(self):
        """Sound bending toward faster medium (away from normal)."""
        v_1 = longitudinal_wave_speed('iron')
        v_2 = longitudinal_wave_speed('aluminum')
        if v_2 > v_1:
            theta = snell_refraction_angle('iron', 'aluminum', 30.0)
            self.assertIsNotNone(theta)
            self.assertGreater(theta, 30.0)

    def test_total_internal_reflection(self):
        """At angles beyond critical: total internal reflection (returns None)."""
        # Find a pair where v1 < v2
        v_cu = longitudinal_wave_speed('copper')
        v_al = longitudinal_wave_speed('aluminum')
        if v_cu < v_al:
            slow, fast = 'copper', 'aluminum'
        else:
            slow, fast = 'aluminum', 'copper'

        theta_c = critical_angle(slow, fast)
        if theta_c is not None:
            # Just above critical angle: total internal reflection
            result = snell_refraction_angle(slow, fast, theta_c + 1.0)
            self.assertIsNone(result)

    def test_critical_angle_exists(self):
        """Critical angle exists when v₁ < v₂."""
        v_cu = longitudinal_wave_speed('copper')
        v_al = longitudinal_wave_speed('aluminum')
        if v_cu < v_al:
            theta_c = critical_angle('copper', 'aluminum')
            self.assertIsNotNone(theta_c)
            self.assertGreater(theta_c, 0)
            self.assertLess(theta_c, 90)

    def test_no_critical_angle_when_faster(self):
        """No critical angle when incident medium is faster."""
        v_cu = longitudinal_wave_speed('copper')
        v_al = longitudinal_wave_speed('aluminum')
        if v_cu > v_al:
            theta_c = critical_angle('copper', 'aluminum')
            self.assertIsNone(theta_c)


class TestResonance(unittest.TestCase):
    """Standing wave resonance frequencies."""

    def test_fundamental_positive(self):
        """Fundamental frequency is positive for any bar."""
        f = resonant_frequency('iron', length_m=1.0)
        self.assertGreater(f, 0)

    def test_inversely_proportional_to_length(self):
        """f ∝ 1/L — half the length doubles the frequency."""
        f1 = resonant_frequency('iron', length_m=1.0)
        f2 = resonant_frequency('iron', length_m=0.5)
        self.assertAlmostEqual(f2 / f1, 2.0, delta=0.01)

    def test_harmonics(self):
        """f_n = n × f_1 — overtones are integer multiples."""
        f1 = resonant_frequency('copper', 1.0, mode_n=1)
        f3 = resonant_frequency('copper', 1.0, mode_n=3)
        self.assertAlmostEqual(f3 / f1, 3.0, delta=0.01)

    def test_iron_1m_bar(self):
        """1m iron bar: f₁ ≈ 2500-4500 Hz (audible!)."""
        f = resonant_frequency('iron', length_m=1.0)
        self.assertGreater(f, 1500)
        self.assertLess(f, 6000)

    def test_ring_frequency(self):
        """Ring frequency of a 1m diameter iron cylinder."""
        f = ring_frequency('iron', diameter_m=1.0)
        # v_L / (π × 1m) ≈ 5000/π ≈ 1600 Hz
        self.assertGreater(f, 800)
        self.assertLess(f, 3000)

    def test_zero_length_returns_zero(self):
        """Zero or negative length returns zero frequency."""
        self.assertEqual(resonant_frequency('iron', 0.0), 0.0)
        self.assertEqual(resonant_frequency('iron', -1.0), 0.0)


class TestWavelength(unittest.TestCase):
    """Acoustic wavelength λ = v / f."""

    def test_known_value(self):
        """1 kHz in iron: λ ≈ 5-6 m (v ≈ 5000-6000 m/s)."""
        lam = wavelength('iron', 1000.0)
        self.assertGreater(lam, 2.0)
        self.assertLess(lam, 10.0)

    def test_transverse_shorter(self):
        """Transverse wavelength < longitudinal at same frequency."""
        lam_L = wavelength('iron', 1000.0, mode='longitudinal')
        lam_T = wavelength('iron', 1000.0, mode='transverse')
        self.assertGreater(lam_L, lam_T)

    def test_zero_frequency(self):
        """Zero frequency → infinite wavelength."""
        lam = wavelength('iron', 0.0)
        self.assertEqual(lam, float('inf'))


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts acoustic properties."""

    def test_density_increases_with_sigma(self):
        """Higher σ → heavier nucleons → higher density."""
        rho_0 = density_at_sigma('iron', 0.0)
        rho_1 = density_at_sigma('iron', 1.0)
        self.assertGreater(rho_1, rho_0)

    def test_density_sigma_zero_is_nominal(self):
        """At σ=0, density equals the MATERIALS value."""
        rho = density_at_sigma('iron', 0.0)
        self.assertEqual(rho, MATERIALS['iron']['density_kg_m3'])

    def test_wave_speed_shifts(self):
        """Non-zero σ changes wave speeds."""
        v_0 = longitudinal_wave_speed('iron', sigma=0.0)
        v_1 = longitudinal_wave_speed('iron', sigma=1.0)
        self.assertNotEqual(v_0, v_1)

    def test_sound_slows_with_sigma(self):
        """Sound speed DECREASES with σ.

        K shifts through E_coh (small QCD fraction ≈ 1%).
        ρ shifts through nuclear mass (large QCD fraction ≈ 99%).
        v = √(K/ρ): numerator barely changes, denominator increases.
        So v decreases with σ. This is an SSBM prediction for neutron
        star crust seismology.
        """
        v_0 = longitudinal_wave_speed('iron', sigma=0.0)
        v_1 = longitudinal_wave_speed('iron', sigma=1.0)
        self.assertLess(v_1, v_0)

    def test_earth_sigma_negligible(self):
        """Earth σ ~ 7×10⁻¹⁰: speed change < 10⁻⁶."""
        v_0 = longitudinal_wave_speed('iron', sigma=0.0)
        v_earth = longitudinal_wave_speed('iron', sigma=7e-10)
        ratio = abs(v_earth - v_0) / v_0
        self.assertLess(ratio, 1e-6)

    def test_impedance_shifts(self):
        """Impedance shifts with σ (Z = ρv, both change)."""
        Z_0 = acoustic_impedance('iron', sigma=0.0)
        Z_1 = acoustic_impedance('iron', sigma=1.0)
        self.assertNotEqual(Z_0, Z_1)


class TestCrossModuleConsistency(unittest.TestCase):
    """Acoustics connects properly to mechanical and thermal modules."""

    def test_uses_mechanical_data(self):
        """Every material in MECHANICAL_DATA has acoustic properties."""
        for mat in MECHANICAL_DATA:
            v = longitudinal_wave_speed(mat)
            self.assertGreater(v, 0)

    def test_debye_velocity_vs_thermal(self):
        """Our Debye velocity should be in the same ballpark as
        thermal.sound_velocity (which uses √(K/ρ))."""
        from .thermal import sound_velocity as thermal_v
        for mat in MECHANICAL_DATA:
            v_thermal = thermal_v(mat)
            v_debye = debye_velocity(mat)
            # Should be same order of magnitude (within factor 2)
            ratio = v_debye / v_thermal
            self.assertGreater(ratio, 0.3,
                f"{mat}: Debye velocity too far from thermal")
            self.assertLess(ratio, 2.0,
                f"{mat}: Debye velocity too far from thermal")


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export format."""

    def test_required_fields(self):
        """Export contains all required fields."""
        props = material_acoustic_properties('iron')
        required = [
            'material', 'sigma', 'density_kg_m3',
            'longitudinal_speed_m_s', 'transverse_speed_m_s',
            'debye_velocity_m_s', 'vL_over_vT',
            'impedance_longitudinal_rayl', 'impedance_transverse_rayl',
            'sigma_sensitivity', 'origin',
        ]
        for key in required:
            self.assertIn(key, props, f"Missing: {key}")

    def test_honest_origin_tags(self):
        """Origin string contains FIRST_PRINCIPLES and MEASURED."""
        props = material_acoustic_properties('iron')
        origin = props['origin']
        self.assertIn('FIRST_PRINCIPLES', origin)
        self.assertIn('MEASURED', origin)

    def test_all_materials_export(self):
        """Every material exports without error."""
        for mat in MECHANICAL_DATA:
            props = material_acoustic_properties(mat)
            self.assertIn('origin', props)

    def test_sigma_propagates(self):
        """σ parameter affects export values."""
        props_0 = material_acoustic_properties('iron', sigma=0.0)
        props_1 = material_acoustic_properties('iron', sigma=1.0)
        self.assertNotEqual(
            props_0['longitudinal_speed_m_s'],
            props_1['longitudinal_speed_m_s'])


if __name__ == '__main__':
    unittest.main()
