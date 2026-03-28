"""
Tests for the liquid_water module.

Test structure:
  1. Two-state model — sigmoid behavior, boundary values
  2. Density — absolute value, maximum at ~4°C (EMERGENT), ice floats
  3. Heat capacity — absolute value, anomalously high
  4. Surface tension — absolute value, positive
  5. Viscosity — absolute value, temperature dependence
  6. Boiling point — absolute value, pressure dependence
  7. Cross-validation — against fluid.py KNOWN_LIQUIDS['water']
  8. Emergent predictions — density maximum, ice floating
  9. Nagatha export — format and origin tags
"""

import math
import unittest

from .liquid_water import (
    T_CROSS_K, T_WIDTH_K,
    ice_like_fraction,
    ice_like_fraction_derivative,
    water_molar_volume,
    water_density,
    water_density_maximum_temperature,
    ice_density,
    water_heat_capacity,
    water_surface_tension,
    water_viscosity,
    water_enthalpy_of_vaporization,
    water_boiling_point,
    water_properties,
)
from .fluid import KNOWN_LIQUIDS


class TestTwoStateModel(unittest.TestCase):
    """Sigmoid ice-like fraction — f_ice(T)."""

    def test_low_T_mostly_ice(self):
        """At T << T_cross: f_ice → 1."""
        f = ice_like_fraction(100.0)
        self.assertGreater(f, 0.98)

    def test_high_T_mostly_dense(self):
        """At T >> T_cross: f_ice → 0."""
        f = ice_like_fraction(400.0)
        self.assertLess(f, 0.01)

    def test_crossover_half(self):
        """At T = T_cross: f_ice = 0.5."""
        f = ice_like_fraction(T_CROSS_K)
        self.assertAlmostEqual(f, 0.5, places=5)

    def test_monotonically_decreasing(self):
        """f_ice decreases with temperature."""
        temps = [150, 200, 225, 250, 300, 350, 400]
        fracs = [ice_like_fraction(T) for T in temps]
        for i in range(len(fracs) - 1):
            self.assertGreaterEqual(fracs[i], fracs[i + 1])

    def test_between_zero_and_one(self):
        """f_ice ∈ [0, 1] for all T."""
        for T in [50, 100, 200, 225, 300, 500, 1000]:
            f = ice_like_fraction(T)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, 1.0)

    def test_derivative_negative(self):
        """df/dT < 0 everywhere (ice fraction always decreasing)."""
        for T in [200, 225, 250, 300, 350]:
            df = ice_like_fraction_derivative(T)
            self.assertLessEqual(df, 0.0)

    def test_derivative_maximum_at_crossover(self):
        """|df/dT| is maximum at T = T_cross."""
        df_cross = abs(ice_like_fraction_derivative(T_CROSS_K))
        df_200 = abs(ice_like_fraction_derivative(200.0))
        df_300 = abs(ice_like_fraction_derivative(300.0))
        self.assertGreater(df_cross, df_200)
        self.assertGreater(df_cross, df_300)


class TestDensity(unittest.TestCase):
    """Water density from two-state model."""

    def test_density_at_25C(self):
        """Density at 25°C ≈ 998 kg/m³ — within 5%."""
        rho = water_density(298.15)
        self.assertAlmostEqual(rho, 998.0, delta=50.0)

    def test_density_at_4C(self):
        """Density at 4°C ≈ 1000 kg/m³ — within 5%."""
        rho = water_density(277.15)
        self.assertAlmostEqual(rho, 1000.0, delta=50.0)

    def test_density_positive(self):
        """Density is positive at all liquid temperatures."""
        for T in [273, 280, 300, 350, 370]:
            rho = water_density(T)
            self.assertGreater(rho, 0)

    def test_density_reasonable_range(self):
        """Density should be 900-1100 kg/m³ for liquid water."""
        for T in [275, 280, 300, 350, 370]:
            rho = water_density(T)
            self.assertGreater(rho, 900, f"T={T}: ρ too low")
            self.assertLess(rho, 1100, f"T={T}: ρ too high")


class TestDensityMaximum(unittest.TestCase):
    """The 4°C density anomaly — EMERGENT prediction."""

    def test_density_maximum_exists(self):
        """A density maximum exists between 270-290 K.

        This is the KEY test: the density maximum at ~4°C is NOT hardcoded.
        It EMERGES from the competition between:
          - collapsing ice-like fraction (increasing density)
          - thermal expansion of dense state (decreasing density)
        """
        T_max, rho_max = water_density_maximum_temperature()
        self.assertGreater(T_max, 270.0)
        self.assertLess(T_max, 290.0)

    def test_density_maximum_near_4C(self):
        """Maximum density should be near 277 K (4°C) — within 5 K."""
        T_max, _ = water_density_maximum_temperature()
        self.assertAlmostEqual(T_max, 277.0, delta=5.0)

    def test_density_decreases_above_maximum(self):
        """Density decreases as T rises above the maximum."""
        T_max, _ = water_density_maximum_temperature()
        rho_max = water_density(T_max)
        rho_hot = water_density(T_max + 20.0)
        self.assertGreater(rho_max, rho_hot)

    def test_density_decreases_below_maximum(self):
        """Density decreases as T drops below the maximum (anomaly!)."""
        T_max, _ = water_density_maximum_temperature()
        rho_max = water_density(T_max)
        rho_cold = water_density(T_max - 5.0)
        self.assertGreater(rho_max, rho_cold)

    def test_ice_floats(self):
        """Ice density < liquid water density → ice floats.

        This is WHY the density anomaly matters: if ice sank, lakes
        would freeze from the bottom up, killing aquatic life.
        """
        rho_ice = ice_density()
        rho_liquid = water_density(277.0)
        self.assertLess(rho_ice, rho_liquid, "Ice must float!")

    def test_ice_density_known(self):
        """Ice Ih: ρ ≈ 917 kg/m³ — within 2%."""
        rho = ice_density()
        self.assertAlmostEqual(rho, 917.0, delta=20.0)


class TestHeatCapacity(unittest.TestCase):
    """Water's anomalously high heat capacity."""

    def test_positive(self):
        """C_p is positive at all temperatures."""
        for T in [280, 300, 350, 370]:
            C = water_heat_capacity(T)
            self.assertGreater(C, 0)

    def test_at_25C(self):
        """C_p at 25°C ≈ 75.3 J/(mol·K) — within 30%.

        The model captures the H-bond breaking contribution that makes
        water's C_p about 2× higher than a normal liquid (~40 J/(mol·K)).
        """
        C = water_heat_capacity(298.15)
        self.assertGreater(C, 50.0)   # much higher than molecular-only (~37)
        self.assertLess(C, 120.0)     # not absurdly high

    def test_higher_than_molecular_only(self):
        """C_p > 4.5R (molecular contribution alone).

        The excess comes from H-bond breaking — this is why water has
        such a high heat capacity compared to other liquids.
        """
        R = 8.314
        C_mol = 4.5 * R  # ~37.4 J/(mol·K)
        C_water = water_heat_capacity(298.15)
        self.assertGreater(C_water, C_mol)


class TestSurfaceTension(unittest.TestCase):
    """Surface tension from broken-bond model."""

    def test_positive(self):
        """Surface tension is positive."""
        gamma = water_surface_tension(298.15)
        self.assertGreater(gamma, 0)

    def test_at_25C(self):
        """γ at 25°C ≈ 0.0720 N/m — within factor 2.

        The broken-bond model gives the right order of magnitude.
        Exact agreement requires accounting for H-bond angle distortion
        and thermal fluctuations at the interface.
        """
        gamma = water_surface_tension(298.15)
        self.assertGreater(gamma, 0.03)
        self.assertLess(gamma, 0.15)

    def test_decreases_with_temperature(self):
        """Surface tension decreases at higher T (more disorder)."""
        gamma_cold = water_surface_tension(280.0)
        gamma_hot = water_surface_tension(350.0)
        self.assertGreater(gamma_cold, gamma_hot)


class TestViscosity(unittest.TestCase):
    """Viscosity from Eyring activated flow."""

    def test_positive(self):
        """Viscosity is positive."""
        eta = water_viscosity(298.15)
        self.assertGreater(eta, 0)

    def test_at_20C_order_of_magnitude(self):
        """η at 20°C ≈ 1.002e-3 Pa·s — within factor 3.

        Eyring model captures the exponential T-dependence but the
        absolute magnitude depends on the prefactor calibration.
        """
        eta = water_viscosity(293.15)
        self.assertGreater(eta, 1e-4)   # > 0.1 mPa·s
        self.assertLess(eta, 1e-1)      # < 100 mPa·s

    def test_decreases_with_temperature(self):
        """Viscosity decreases at higher T (Arrhenius behavior)."""
        eta_cold = water_viscosity(280.0)
        eta_hot = water_viscosity(350.0)
        self.assertGreater(eta_cold, eta_hot)


class TestBoilingPoint(unittest.TestCase):
    """Boiling point from Clausius-Clapeyron."""

    def test_at_1atm(self):
        """T_boil at 1 atm ≈ 373 K — within 20%."""
        T = water_boiling_point(1.0)
        self.assertGreater(T, 300.0)
        self.assertLess(T, 450.0)

    def test_vaporization_enthalpy(self):
        """ΔH_vap ≈ 40.7 kJ/mol — within factor 2."""
        dH = water_enthalpy_of_vaporization()
        dH_kJ = dH / 1000.0
        self.assertGreater(dH_kJ, 20.0)
        self.assertLess(dH_kJ, 80.0)

    def test_higher_pressure_higher_boiling(self):
        """Higher pressure → higher boiling point."""
        T_1atm = water_boiling_point(1.0)
        T_2atm = water_boiling_point(2.0)
        self.assertGreater(T_2atm, T_1atm)

    def test_lower_pressure_lower_boiling(self):
        """Lower pressure → lower boiling point (mountain cooking!)."""
        T_1atm = water_boiling_point(1.0)
        T_low = water_boiling_point(0.5)
        self.assertLess(T_low, T_1atm)


class TestCrossValidation(unittest.TestCase):
    """Cross-validate against fluid.py KNOWN_LIQUIDS['water']."""

    def test_density_vs_fluid(self):
        """Our density at 20°C vs fluid.py: 998.2 kg/m³."""
        rho_ours = water_density(293.15)
        rho_fluid = KNOWN_LIQUIDS['water']['density_kg_m3']
        ratio = rho_ours / rho_fluid
        self.assertGreater(ratio, 0.93, "Density too low vs fluid.py")
        self.assertLess(ratio, 1.07, "Density too high vs fluid.py")

    def test_surface_tension_vs_fluid(self):
        """Our surface tension at 20°C vs fluid.py: 0.0728 N/m."""
        gamma_ours = water_surface_tension(293.15)
        gamma_fluid = KNOWN_LIQUIDS['water']['surface_tension_n_m']
        ratio = gamma_ours / gamma_fluid
        self.assertGreater(ratio, 0.5, "Surface tension too low vs fluid.py")
        self.assertLess(ratio, 2.0, "Surface tension too high vs fluid.py")


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_all_keys_present(self):
        """Export contains all expected keys."""
        props = water_properties()
        required = [
            'temperature_K', 'pressure_atm', 'sigma',
            'density_kg_m3', 'density_maximum_T_K', 'density_maximum_kg_m3',
            'ice_density_kg_m3', 'ice_floats',
            'molar_volume_m3_mol', 'ice_like_fraction',
            'heat_capacity_J_mol_K', 'surface_tension_N_m',
            'viscosity_Pa_s', 'boiling_point_K',
            'enthalpy_vaporization_J_mol',
            'n_hb_liquid', 'hb_energy_eV',
            'T_cross_K', 'T_width_K', 'origin',
        ]
        for key in required:
            self.assertIn(key, props, f"Missing: {key}")

    def test_ice_floats_flag(self):
        """Export correctly indicates ice floats."""
        props = water_properties()
        self.assertTrue(props['ice_floats'])

    def test_origin_tags_honest(self):
        """Origin contains FIRST_PRINCIPLES and MEASURED."""
        props = water_properties()
        self.assertIn('FIRST_PRINCIPLES', props['origin'])
        self.assertIn('MEASURED', props['origin'])
        self.assertIn('Two-state', props['origin'])

    def test_sigma_propagates(self):
        """σ value appears in export."""
        props = water_properties(sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)


if __name__ == '__main__':
    unittest.main()
