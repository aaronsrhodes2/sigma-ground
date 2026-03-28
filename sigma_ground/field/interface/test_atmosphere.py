"""
Tests for atmosphere.py — planetary atmospheric thermodynamics.

Strategy:
  - Test air composition sums to ~1.0
  - Test mean molecular mass ≈ 28.97 g/mol (MEASURED)
  - Test cp, gamma against MEASURED values for dry air
  - Test barometric formula against ISA (International Standard Atmosphere)
  - Test dry adiabatic lapse rate ≈ 9.8 K/km (MEASURED)
  - Test speed of sound ≈ 340 m/s at 15°C (MEASURED)
  - Test Clausius-Clapeyron vapor pressure vs MEASURED
  - Test humidity functions are self-consistent
  - Test moist lapse rate < dry lapse rate
  - Test Rule 9: full_report covers standard conditions

Reference values (MEASURED, ISA/ICAO):
  Mean molar mass dry air: 28.97 g/mol
  cp(dry air, 15°C):      ~29.1 J/(mol·K) = ~1005 J/(kg·K)
  gamma(dry air):          1.400
  Dry adiabatic lapse rate: 9.8 K/km
  Speed of sound (15°C):   340.3 m/s
  P(5500 m):               ~50500 Pa (half atmosphere)
  Scale height (15°C):     ~8500 m
  P_sat(100°C):            101325 Pa (by definition)
  P_sat(20°C):             ~2338 Pa
"""

import math
import unittest

from sigma_ground.field.interface.atmosphere import (
    DRY_AIR_COMPOSITION,
    mean_molecular_mass_kg,
    mean_molar_mass_kg_mol,
    air_cp_molar,
    air_cp_mass,
    air_gamma,
    air_density,
    speed_of_sound,
    scale_height,
    pressure_at_altitude,
    density_at_altitude,
    altitude_for_pressure,
    dry_adiabatic_lapse_rate,
    temperature_at_altitude,
    saturation_vapor_pressure,
    dew_point,
    mixing_ratio,
    absolute_humidity,
    specific_humidity,
    moist_adiabatic_lapse_rate,
    column_number_density,
    atmosphere_report,
    full_report,
)


class TestAirComposition(unittest.TestCase):
    """Dry air composition validation."""

    def test_fractions_sum_near_one(self):
        """Volume fractions should sum to ~1.0."""
        total = sum(DRY_AIR_COMPOSITION.values())
        self.assertAlmostEqual(total, 1.0, delta=0.005)

    def test_N2_dominant(self):
        """N₂ is the dominant component (~78%)."""
        f_N2 = DRY_AIR_COMPOSITION['N2']
        self.assertGreater(f_N2, 0.75)
        self.assertLess(f_N2, 0.80)

    def test_O2_second(self):
        """O₂ is second (~21%)."""
        f_O2 = DRY_AIR_COMPOSITION['O2']
        self.assertGreater(f_O2, 0.20)
        self.assertLess(f_O2, 0.22)

    def test_CO2_ppm(self):
        """CO₂ ≈ 420 ppm."""
        f_CO2 = DRY_AIR_COMPOSITION['CO2']
        ppm = f_CO2 * 1e6
        self.assertGreater(ppm, 350)
        self.assertLess(ppm, 500)


class TestMeanMolecularMass(unittest.TestCase):
    """Mean molecular mass of dry air."""

    def test_molar_mass(self):
        """Mean molar mass ≈ 28.97 g/mol (MEASURED)."""
        M = mean_molar_mass_kg_mol()
        M_gmol = M * 1000
        self.assertGreater(M_gmol, 28.5)
        self.assertLess(M_gmol, 29.5)

    def test_molecular_mass(self):
        """Mean molecular mass ≈ 4.81e-26 kg."""
        m = mean_molecular_mass_kg()
        self.assertGreater(m, 4.5e-26)
        self.assertLess(m, 5.1e-26)

    def test_consistent(self):
        """M_molar = m_molecular × N_A."""
        from sigma_ground.field.constants import N_AVOGADRO
        M = mean_molar_mass_kg_mol()
        m = mean_molecular_mass_kg()
        self.assertAlmostEqual(M, m * N_AVOGADRO, delta=M * 0.01)


class TestHeatCapacity(unittest.TestCase):
    """Air heat capacity and gamma."""

    def test_cp_molar(self):
        """cp(dry air) ≈ 29.1 J/(mol·K) at 288 K."""
        cp = air_cp_molar(288.15)
        self.assertGreater(cp, 28.0)
        self.assertLess(cp, 32.0)

    def test_cp_mass(self):
        """cp(dry air) ≈ 1005 J/(kg·K) at 288 K."""
        cp = air_cp_mass(288.15)
        self.assertGreater(cp, 900)
        self.assertLess(cp, 1100)

    def test_gamma(self):
        """gamma(dry air) ≈ 1.40."""
        g = air_gamma(288.15)
        self.assertGreater(g, 1.35)
        self.assertLess(g, 1.45)

    def test_gamma_decreases_with_T(self):
        """Gamma decreases at high T (vibrational modes activate)."""
        g_low = air_gamma(200.0)
        g_high = air_gamma(1000.0)
        self.assertGreaterEqual(g_low, g_high)


class TestAirDensity(unittest.TestCase):
    """Air density from ideal gas law."""

    def test_sea_level(self):
        """ρ(air, 15°C, 1 atm) ≈ 1.225 kg/m³ (MEASURED, ISA)."""
        rho = air_density(288.15, 101325.0)
        self.assertGreater(rho, 1.1)
        self.assertLess(rho, 1.4)

    def test_decreases_with_T(self):
        """Hotter air is less dense."""
        rho_cold = air_density(250.0)
        rho_hot = air_density(350.0)
        self.assertGreater(rho_cold, rho_hot)

    def test_increases_with_P(self):
        """Higher pressure → denser air."""
        rho_low = air_density(288.15, 50000.0)
        rho_high = air_density(288.15, 101325.0)
        self.assertGreater(rho_high, rho_low)


class TestSpeedOfSound(unittest.TestCase):
    """Speed of sound in air."""

    def test_15C(self):
        """v_sound(15°C) ≈ 340 m/s (MEASURED)."""
        v = speed_of_sound(288.15)
        self.assertGreater(v, 330)
        self.assertLess(v, 350)

    def test_increases_with_T(self):
        """Sound is faster in warmer air."""
        v_cold = speed_of_sound(250.0)
        v_warm = speed_of_sound(310.0)
        self.assertGreater(v_warm, v_cold)

    def test_proportional_to_sqrt_T(self):
        """v ∝ √T (ideal gas)."""
        v1 = speed_of_sound(250.0)
        v2 = speed_of_sound(1000.0)
        ratio = v2 / v1
        expected = math.sqrt(1000.0 / 250.0)
        self.assertAlmostEqual(ratio, expected, delta=0.1)


class TestBarometricFormula(unittest.TestCase):
    """Pressure and density vs altitude."""

    def test_sea_level(self):
        """P(0) = P0."""
        P = pressure_at_altitude(0.0)
        self.assertAlmostEqual(P, 101325.0, delta=1.0)

    def test_half_atmosphere(self):
        """P ≈ P0/2 at ~5500 m."""
        P = pressure_at_altitude(5500.0)
        self.assertGreater(P, 40000)
        self.assertLess(P, 60000)

    def test_decreases_monotonically(self):
        """Pressure decreases with altitude."""
        P_prev = pressure_at_altitude(0.0)
        for z in [1000, 5000, 10000, 20000]:
            P = pressure_at_altitude(z)
            self.assertLess(P, P_prev)
            P_prev = P

    def test_exponential_decay(self):
        """P(z) ≈ P0 × exp(-z/H), so P(H) ≈ P0/e."""
        H = scale_height(288.15)
        P = pressure_at_altitude(H)
        self.assertAlmostEqual(P / 101325.0, 1.0 / math.e, delta=0.05)

    def test_density_decreases(self):
        """Density decreases with altitude."""
        rho_0 = density_at_altitude(0.0)
        rho_5k = density_at_altitude(5000.0)
        self.assertGreater(rho_0, rho_5k)

    def test_altitude_for_pressure_inverse(self):
        """altitude_for_pressure is inverse of pressure_at_altitude."""
        z_test = 3000.0
        P = pressure_at_altitude(z_test)
        z_back = altitude_for_pressure(P)
        self.assertAlmostEqual(z_back, z_test, delta=10.0)


class TestScaleHeight(unittest.TestCase):
    """Atmospheric scale height."""

    def test_value(self):
        """H ≈ 8500 m at 15°C."""
        H = scale_height(288.15)
        self.assertGreater(H, 7500)
        self.assertLess(H, 9500)

    def test_proportional_to_T(self):
        """H ∝ T."""
        H1 = scale_height(250.0)
        H2 = scale_height(500.0)
        self.assertAlmostEqual(H2 / H1, 2.0, delta=0.1)


class TestLapseRate(unittest.TestCase):
    """Adiabatic lapse rates."""

    def test_dry_lapse_rate(self):
        """Γ_d = g/cp ≈ 9.8 K/km (MEASURED)."""
        gamma_d = dry_adiabatic_lapse_rate(288.15)
        gamma_d_km = gamma_d * 1000  # K/m → K/km
        self.assertGreater(gamma_d_km, 8.5)
        self.assertLess(gamma_d_km, 11.0)

    def test_temperature_decreases_with_altitude(self):
        """Temperature drops with altitude (troposphere)."""
        T_0 = temperature_at_altitude(0.0, 288.15)
        T_5k = temperature_at_altitude(5000.0, 288.15)
        self.assertGreater(T_0, T_5k)

    def test_moist_less_than_dry(self):
        """Moist adiabatic lapse rate < dry (latent heat release)."""
        gamma_d = dry_adiabatic_lapse_rate(288.15)
        gamma_m = moist_adiabatic_lapse_rate(288.15)
        self.assertLess(gamma_m, gamma_d)

    def test_moist_positive(self):
        """Moist lapse rate is positive (T still decreases with altitude)."""
        gamma_m = moist_adiabatic_lapse_rate(288.15)
        self.assertGreater(gamma_m, 0)

    def test_moist_approaches_dry_at_low_T(self):
        """At very cold T, little water vapor → moist ≈ dry."""
        gamma_d = dry_adiabatic_lapse_rate(220.0)
        gamma_m = moist_adiabatic_lapse_rate(220.0)
        # Should be within 20% at -53°C
        self.assertAlmostEqual(gamma_m / gamma_d, 1.0, delta=0.2)


class TestVaporPressure(unittest.TestCase):
    """Clausius-Clapeyron saturation vapor pressure."""

    def test_100C(self):
        """P_sat(100°C) ≈ 101325 Pa (boiling point definition)."""
        P = saturation_vapor_pressure(373.15)
        self.assertGreater(P, 80000)
        self.assertLess(P, 130000)

    def test_20C(self):
        """P_sat(20°C) ≈ 2338 Pa (MEASURED)."""
        P = saturation_vapor_pressure(293.15)
        self.assertGreater(P, 1500)
        self.assertLess(P, 4000)

    def test_increases_with_T(self):
        """Vapor pressure increases with temperature."""
        P_cold = saturation_vapor_pressure(273.15)
        P_warm = saturation_vapor_pressure(313.15)
        self.assertGreater(P_warm, P_cold)

    def test_zero_at_zero_T(self):
        """P_sat → 0 as T → 0."""
        P = saturation_vapor_pressure(50.0)
        self.assertLess(P, 1e-10)

    def test_monotonic(self):
        """P_sat is strictly increasing with T."""
        P_prev = 0
        for T in range(200, 400, 10):
            P = saturation_vapor_pressure(float(T))
            self.assertGreater(P, P_prev)
            P_prev = P


class TestHumidity(unittest.TestCase):
    """Humidity functions."""

    def test_dew_point_at_100pct(self):
        """At 100% RH, dew point = air temperature."""
        T = 293.15  # 20°C
        T_dew = dew_point(T, 1.0)
        self.assertAlmostEqual(T_dew, T, delta=1.0)

    def test_dew_point_below_T(self):
        """Dew point < T when RH < 100%."""
        T = 293.15
        T_dew = dew_point(T, 0.5)
        self.assertLess(T_dew, T)

    def test_dew_point_decreases_with_RH(self):
        """Lower RH → lower dew point."""
        T = 293.15
        Td_high = dew_point(T, 0.9)
        Td_low = dew_point(T, 0.3)
        self.assertGreater(Td_high, Td_low)

    def test_mixing_ratio_positive(self):
        """Mixing ratio is positive."""
        w = mixing_ratio(293.15)
        self.assertGreater(w, 0)

    def test_mixing_ratio_increases_with_T(self):
        """Warmer air holds more water."""
        w_cold = mixing_ratio(273.15)
        w_warm = mixing_ratio(313.15)
        self.assertGreater(w_warm, w_cold)

    def test_absolute_humidity_positive(self):
        """Absolute humidity is positive."""
        rho_v = absolute_humidity(293.15)
        self.assertGreater(rho_v, 0)

    def test_specific_humidity_bounded(self):
        """Specific humidity should be between 0 and 1."""
        q = specific_humidity(293.15)
        self.assertGreater(q, 0)
        self.assertLess(q, 1)

    def test_humidity_self_consistent(self):
        """mixing_ratio and specific_humidity should be related: q = w/(1+w)."""
        T = 293.15
        w = mixing_ratio(T)
        q = specific_humidity(T)
        expected_q = w / (1.0 + w)
        self.assertAlmostEqual(q, expected_q, delta=expected_q * 0.1)


class TestColumnDensity(unittest.TestCase):
    """Column number density."""

    def test_positive(self):
        """Column density is positive."""
        N = column_number_density(0.7808)  # N2 fraction
        self.assertGreater(N, 0)

    def test_proportional_to_fraction(self):
        """Column density ∝ volume fraction."""
        N1 = column_number_density(0.5)
        N2 = column_number_density(1.0)
        self.assertAlmostEqual(N2 / N1, 2.0, delta=0.05)


class TestDiagnostics(unittest.TestCase):
    """Report generation."""

    def test_report_complete(self):
        r = atmosphere_report()
        required = [
            'T_surface_K', 'P_surface_Pa',
            'M_air_kg_mol', 'air_density_kg_m3',
            'speed_of_sound_m_s', 'scale_height_m',
            'dry_lapse_rate_K_km',
        ]
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report(self):
        """full_report should return a dict."""
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertGreater(len(r), 0)


if __name__ == '__main__':
    unittest.main()
