"""Tests for grain structure — Hall-Petch, grain growth, microstructure."""

import math
import unittest

from .grain_structure import (
    hall_petch_yield,
    hall_petch_slope,
    grain_size_for_yield,
    grain_growth_rate_constant,
    grain_size_after_anneal,
    time_to_grain_size,
    zener_limit,
    grain_boundary_area_per_volume,
    grain_boundary_energy_density,
    dislocation_density_estimate,
    taylor_hardening_stress,
    polycrystal_yield,
    annealing_profile,
    sigma_hall_petch_shift,
    grain_structure_properties,
    GRAIN_DATA,
)
from .surface import MATERIALS


# ── Rule 9: completeness ────────────────────────────────────────

class TestRule9GrainData(unittest.TestCase):
    """Every material in MATERIALS must have grain structure data."""

    def test_all_materials_have_grain_data(self):
        for key in MATERIALS:
            self.assertIn(key, GRAIN_DATA, f"{key} missing from GRAIN_DATA")

    def test_all_entries_have_required_fields(self):
        required = {
            'sigma_0_Pa', 'k_HP_Pa_sqrtm', 'd_typical_m',
            'Q_gg_eV', 'K0_gg_m2_s', 'd_inverse_HP_m', 'burgers_m',
        }
        for key, data in GRAIN_DATA.items():
            for field in required:
                self.assertIn(field, data, f"{key} missing {field}")

    def test_sigma_0_positive(self):
        for key, data in GRAIN_DATA.items():
            self.assertGreater(data['sigma_0_Pa'], 0, f"{key}: σ_0 must be > 0")

    def test_k_HP_non_negative(self):
        for key, data in GRAIN_DATA.items():
            self.assertGreaterEqual(data['k_HP_Pa_sqrtm'], 0,
                                    f"{key}: k_HP must be >= 0")

    def test_burgers_vector_plausible(self):
        """Burgers vector should be ~2-4 Å for crystalline materials."""
        for key, data in GRAIN_DATA.items():
            b = data['burgers_m']
            # Amorphous materials have no crystalline slip system → b = 0
            if b == 0 or MATERIALS.get(key, {}).get('crystal_structure') == 'amorphous':
                continue
            self.assertGreater(b, 1e-10, f"{key}: b too small")
            self.assertLess(b, 5e-10, f"{key}: b too large")


# ── Hall-Petch ───────────────────────────────────────────────────

class TestHallPetch(unittest.TestCase):

    def test_positive_for_all(self):
        for key in GRAIN_DATA:
            d = GRAIN_DATA[key]['d_typical_m']
            sy = hall_petch_yield(key, d)
            self.assertGreater(sy, 0, f"{key}: HP yield must be > 0")

    def test_smaller_grains_stronger(self):
        """Finer grain size → higher yield stress (Hall-Petch)."""
        for key, data in GRAIN_DATA.items():
            if data['k_HP_Pa_sqrtm'] > 0:
                sy_fine = hall_petch_yield(key, 10e-6)  # 10 μm
                sy_coarse = hall_petch_yield(key, 100e-6)  # 100 μm
                self.assertGreater(sy_fine, sy_coarse,
                                   f"{key}: fine grains should be stronger")

    def test_approaches_sigma_0_at_large_d(self):
        """At very large grain size, yield → σ_0."""
        for key, data in GRAIN_DATA.items():
            sy = hall_petch_yield(key, 1.0)  # 1 meter grain
            sigma_0 = data['sigma_0_Pa']
            # Should be very close to σ_0
            self.assertAlmostEqual(sy / sigma_0, 1.0, places=1)

    def test_iron_stronger_hp_than_aluminum(self):
        """Iron (BCC) has stronger HP effect than Al (FCC)."""
        k_Fe = GRAIN_DATA['iron']['k_HP_Pa_sqrtm']
        k_Al = GRAIN_DATA['aluminum']['k_HP_Pa_sqrtm']
        self.assertGreater(k_Fe, k_Al)

    def test_inverse_hall_petch(self):
        """Below critical grain size, yield should decrease."""
        for key, data in GRAIN_DATA.items():
            d_inv = data['d_inverse_HP_m']
            if d_inv > 0 and data['k_HP_Pa_sqrtm'] > 0:
                sy_crit = hall_petch_yield(key, d_inv)
                sy_nano = hall_petch_yield(key, d_inv / 2.0)
                self.assertLess(sy_nano, sy_crit,
                                f"{key}: inverse HP below d_crit")

    def test_silicon_no_hp_effect(self):
        """Silicon (brittle) should have k_HP = 0."""
        self.assertEqual(GRAIN_DATA['silicon']['k_HP_Pa_sqrtm'], 0.0)
        # Yield should just be σ_0 regardless of grain size
        sy1 = hall_petch_yield('silicon', 10e-6)
        sy2 = hall_petch_yield('silicon', 100e-6)
        self.assertAlmostEqual(sy1, sy2)


# ── Grain size for yield ─────────────────────────────────────────

class TestGrainSizeForYield(unittest.TestCase):

    def test_round_trip(self):
        """grain_size_for_yield(HP_yield(d)) should return d."""
        for key, data in GRAIN_DATA.items():
            if data['k_HP_Pa_sqrtm'] > 0:
                d = 25e-6  # 25 μm
                sy = hall_petch_yield(key, d)
                d_calc = grain_size_for_yield(key, sy)
                self.assertAlmostEqual(d_calc, d, places=10)

    def test_below_sigma_0_returns_inf(self):
        d = grain_size_for_yield('iron', 10e6)  # Below σ_0
        self.assertEqual(d, float('inf'))

    def test_higher_target_smaller_grain(self):
        d1 = grain_size_for_yield('iron', 300e6)
        d2 = grain_size_for_yield('iron', 500e6)
        self.assertGreater(d1, d2)


# ── Grain Growth ─────────────────────────────────────────────────

class TestGrainGrowth(unittest.TestCase):

    def test_rate_constant_positive_at_high_T(self):
        for key in GRAIN_DATA:
            K_g = grain_growth_rate_constant(key, 1000.0)
            self.assertGreater(K_g, 0)

    def test_rate_increases_with_temperature(self):
        for key in GRAIN_DATA:
            K_low = grain_growth_rate_constant(key, 500.0)
            K_high = grain_growth_rate_constant(key, 1000.0)
            self.assertGreater(K_high, K_low,
                               f"{key}: growth should be faster at higher T")

    def test_grains_grow_with_time(self):
        d0 = 10e-6
        d1 = grain_size_after_anneal('copper', d0, 800.0, 3600.0)
        self.assertGreater(d1, d0)

    def test_no_growth_at_zero_time(self):
        d0 = 10e-6
        d = grain_size_after_anneal('iron', d0, 1000.0, 0.0)
        self.assertAlmostEqual(d, d0)

    def test_parabolic_growth(self):
        """d² should increase linearly with time."""
        d0 = 10e-6
        T = 900.0
        d1 = grain_size_after_anneal('copper', d0, T, 1000.0)
        d2 = grain_size_after_anneal('copper', d0, T, 2000.0)
        # d²-d₀² should double when time doubles
        delta1 = d1 ** 2 - d0 ** 2
        delta2 = d2 ** 2 - d0 ** 2
        self.assertAlmostEqual(delta2 / delta1, 2.0, places=5)

    def test_time_to_grain_size_round_trip(self):
        d0 = 10e-6
        T = 800.0
        t = 3600.0
        d_final = grain_size_after_anneal('nickel', d0, T, t)
        t_calc = time_to_grain_size('nickel', d0, d_final, T)
        self.assertAlmostEqual(t_calc, t, places=3)

    def test_tungsten_grows_slowly(self):
        """Tungsten (refractory) should grow much slower than aluminum."""
        d0 = 10e-6
        d_W = grain_size_after_anneal('tungsten', d0, 800.0, 3600.0)
        d_Al = grain_size_after_anneal('aluminum', d0, 800.0, 3600.0)
        self.assertGreater(d_Al, d_W)


# ── Zener Pinning ────────────────────────────────────────────────

class TestZenerPinning(unittest.TestCase):

    def test_positive(self):
        d_max = zener_limit(50e-9, 0.05)
        self.assertGreater(d_max, 0)

    def test_more_particles_smaller_limit(self):
        d1 = zener_limit(50e-9, 0.01)
        d2 = zener_limit(50e-9, 0.05)
        self.assertGreater(d1, d2)

    def test_larger_particles_larger_limit(self):
        d1 = zener_limit(10e-9, 0.05)
        d2 = zener_limit(50e-9, 0.05)
        self.assertGreater(d2, d1)

    def test_zero_fraction_infinite(self):
        self.assertEqual(zener_limit(50e-9, 0.0), float('inf'))

    def test_known_value(self):
        """d_max = 4r/(3f). r=50nm, f=0.05 → d_max = 1.33 μm."""
        d = zener_limit(50e-9, 0.05)
        expected = 4 * 50e-9 / (3 * 0.05)
        self.assertAlmostEqual(d, expected)


# ── Grain boundary area and energy ──────────────────────────────

class TestGrainBoundary(unittest.TestCase):

    def test_area_per_volume_positive(self):
        S_v = grain_boundary_area_per_volume(25e-6)
        self.assertGreater(S_v, 0)

    def test_smaller_grains_more_boundary(self):
        S_fine = grain_boundary_area_per_volume(10e-6)
        S_coarse = grain_boundary_area_per_volume(100e-6)
        self.assertGreater(S_fine, S_coarse)

    def test_energy_density_positive(self):
        E = grain_boundary_energy_density('iron', 25e-6)
        self.assertGreater(E, 0)


# ── Dislocation density and Taylor hardening ─────────────────────

class TestDislocationDensity(unittest.TestCase):

    def test_annealed_baseline(self):
        rho = dislocation_density_estimate('iron', 0.0)
        self.assertAlmostEqual(rho, 1e10)  # Annealed baseline

    def test_increases_with_strain(self):
        rho_0 = dislocation_density_estimate('copper', 0.0)
        rho_1 = dislocation_density_estimate('copper', 0.10)
        self.assertGreater(rho_1, rho_0)

    def test_cold_worked_high_density(self):
        """At large strain, ρ should be >> 10¹⁰."""
        rho = dislocation_density_estimate('iron', 0.50)
        self.assertGreater(rho, 1e13)


class TestTaylorHardening(unittest.TestCase):

    def test_positive(self):
        sigma_t = taylor_hardening_stress('iron', 1e14)
        self.assertGreater(sigma_t, 0)

    def test_increases_with_density(self):
        s1 = taylor_hardening_stress('copper', 1e12)
        s2 = taylor_hardening_stress('copper', 1e14)
        self.assertGreater(s2, s1)

    def test_sqrt_dependence(self):
        """σ ∝ √ρ, so 100× density → 10× stress."""
        s1 = taylor_hardening_stress('iron', 1e12)
        s2 = taylor_hardening_stress('iron', 1e14)
        ratio = s2 / s1
        self.assertAlmostEqual(ratio, 10.0, places=3)


# ── Combined polycrystal yield ──────────────────────────────────

class TestPolycrystalYield(unittest.TestCase):

    def test_greater_than_hp_alone(self):
        """With plastic strain, should exceed pure HP yield."""
        d = 25e-6
        sy_hp = hall_petch_yield('iron', d)
        sy_combined = polycrystal_yield('iron', d, plastic_strain=0.10)
        self.assertGreater(sy_combined, sy_hp)

    def test_no_strain_equals_hp(self):
        """At zero plastic strain, should equal HP yield."""
        d = 25e-6
        sy_hp = hall_petch_yield('copper', d)
        sy_pc = polycrystal_yield('copper', d, 0.0)
        self.assertAlmostEqual(sy_pc, sy_hp)


# ── Annealing profile ──────────────────────────────────────────

class TestAnnealingProfile(unittest.TestCase):

    def test_correct_length(self):
        profile = annealing_profile('copper', 10e-6, 800.0, 3600.0, steps=20)
        self.assertEqual(len(profile), 21)

    def test_grain_size_increases(self):
        profile = annealing_profile('aluminum', 10e-6, 700.0, 7200.0, steps=10)
        for i in range(1, len(profile)):
            self.assertGreaterEqual(profile[i]['grain_size_m'],
                                    profile[i - 1]['grain_size_m'])

    def test_yield_stress_decreases(self):
        """As grains grow, yield stress should decrease (for k_HP > 0)."""
        profile = annealing_profile('iron', 10e-6, 1000.0, 36000.0, steps=10)
        # Only check if grains actually grew
        if profile[-1]['grain_size_m'] > profile[0]['grain_size_m']:
            self.assertLess(profile[-1]['yield_stress_pa'],
                            profile[0]['yield_stress_pa'])


# ── σ-field coupling ────────────────────────────────────────────

class TestSigmaGrainStructure(unittest.TestCase):

    def test_identity_at_zero(self):
        for key in GRAIN_DATA:
            d = GRAIN_DATA[key]['d_typical_m']
            self.assertAlmostEqual(sigma_hall_petch_shift(key, d, 0.0), 1.0)

    def test_positive_sigma_strengthens(self):
        ratio = sigma_hall_petch_shift('iron', 25e-6, 0.01)
        self.assertGreater(ratio, 1.0)

    def test_sigma_slows_grain_growth(self):
        """Positive σ increases Q_gg → slower growth."""
        d0 = 10e-6
        d_0 = grain_size_after_anneal('copper', d0, 800.0, 3600.0, 0.0)
        d_s = grain_size_after_anneal('copper', d0, 800.0, 3600.0, 0.01)
        self.assertLess(d_s, d_0)  # Slower growth at positive σ


# ── Nagatha export ──────────────────────────────────────────────

class TestGrainStructureProperties(unittest.TestCase):

    def test_all_materials_export(self):
        for key in GRAIN_DATA:
            props = grain_structure_properties(key)
            self.assertEqual(props['material'], key)
            self.assertGreater(props['yield_stress_pa'], 0)
            self.assertIn('origin', props)

    def test_expected_keys(self):
        props = grain_structure_properties('iron')
        expected = {
            'material', 'sigma', 'grain_size_m', 'sigma_0_Pa',
            'k_HP_Pa_sqrtm', 'yield_stress_pa',
            'gb_area_per_volume_1_m', 'gb_energy_density_J_m3',
            'dislocation_density_annealed_1_m2', 'burgers_vector_m',
            'd_inverse_HP_m', 'sigma_hp_ratio', 'origin',
        }
        for k in expected:
            self.assertIn(k, props, f"Missing key: {k}")

    def test_custom_grain_size(self):
        props = grain_structure_properties('iron', grain_size_m=5e-6)
        self.assertAlmostEqual(props['grain_size_m'], 5e-6)


if __name__ == '__main__':
    unittest.main()
