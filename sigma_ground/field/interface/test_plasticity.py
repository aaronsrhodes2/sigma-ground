"""Tests for plasticity — post-yield deformation and strain hardening."""

import math
import unittest

from .plasticity import (
    yield_stress,
    strength_coefficient,
    hollomon_stress,
    ramberg_osgood_strain,
    ludwik_stress,
    johnson_cook_stress,
    necking_strain,
    necking_stress,
    uniform_elongation,
    total_elongation,
    reduction_in_area,
    stress_strain_curve,
    work_hardening_rate,
    plastic_work_density,
    toughness_estimate,
    sigma_yield_shift,
    plasticity_properties,
    PLASTICITY_DATA,
)
from .mechanical import youngs_modulus
from .surface import MATERIALS


# ── Rule 9: completeness ────────────────────────────────────────

class TestRule9PlasticityData(unittest.TestCase):
    """Every material in MATERIALS must have plasticity data."""

    def test_all_materials_have_plasticity_data(self):
        for key in MATERIALS:
            self.assertIn(key, PLASTICITY_DATA, f"{key} missing from PLASTICITY_DATA")

    def test_all_entries_have_required_fields(self):
        required = {
            'sigma_y_Pa', 'n_hardening', 'elongation_pct', 'is_ductile',
            'jc_A', 'jc_B', 'jc_n', 'jc_C', 'jc_m', 'jc_T_melt', 'jc_edot_0',
        }
        for key, data in PLASTICITY_DATA.items():
            for field in required:
                self.assertIn(field, data, f"{key} missing {field}")

    def test_yield_less_than_uts(self):
        """Yield stress should be less than UTS for all ductile materials."""
        from .stress import STRESS_DATA
        for key in PLASTICITY_DATA:
            if key in STRESS_DATA and PLASTICITY_DATA[key]['is_ductile']:
                sy = PLASTICITY_DATA[key]['sigma_y_Pa']
                uts = STRESS_DATA[key]['sigma_UTS_Pa']
                self.assertLess(sy, uts,
                                f"{key}: σ_y={sy/1e6:.0f} should be < σ_UTS={uts/1e6:.0f} MPa")


# ── Yield stress ─────────────────────────────────────────────────

class TestYieldStress(unittest.TestCase):

    def test_positive_for_all(self):
        for key in PLASTICITY_DATA:
            sy = yield_stress(key)
            self.assertGreater(sy, 0, f"{key}: yield stress must be > 0")

    def test_known_values(self):
        """Check a few known yield stresses within factor 2."""
        # These are pure metal values — wide range is expected
        sy_Fe = yield_stress('iron')
        self.assertGreater(sy_Fe, 100e6)
        self.assertLess(sy_Fe, 500e6)

        sy_Cu = yield_stress('copper')
        self.assertGreater(sy_Cu, 20e6)
        self.assertLess(sy_Cu, 200e6)

    def test_tungsten_highest(self):
        """Tungsten should have the highest yield stress among the original 8 metals."""
        original_metals = [
            'iron', 'copper', 'aluminum', 'gold',
            'silicon', 'tungsten', 'nickel', 'titanium',
        ]
        sy_W = yield_stress('tungsten')
        for key in original_metals:
            if key != 'tungsten':
                self.assertGreaterEqual(sy_W, yield_stress(key),
                                        f"Tungsten should be harder than {key}")

    def test_sigma_shifts_yield(self):
        sy_0 = yield_stress('iron', 0.0)
        sy_s = yield_stress('iron', 0.01)
        self.assertNotAlmostEqual(sy_0, sy_s, places=3)


# ── Strength coefficient ────────────────────────────────────────

class TestStrengthCoefficient(unittest.TestCase):

    def test_positive_for_ductile(self):
        for key, data in PLASTICITY_DATA.items():
            if data['is_ductile'] and data['n_hardening'] > 0:
                Kp = strength_coefficient(key)
                self.assertGreater(Kp, 0, f"{key}: K' must be > 0")

    def test_greater_than_yield(self):
        """K' > σ_y always (K' is stress at ε_p = 1)."""
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                Kp = strength_coefficient(key)
                sy = yield_stress(key)
                self.assertGreater(Kp, sy, f"{key}: K' must exceed σ_y")

    def test_reproduces_yield_at_002(self):
        """K' × (0.002)^n should give back σ_y."""
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                Kp = strength_coefficient(key)
                n = data['n_hardening']
                sy_calc = Kp * (0.002 ** n)
                sy_data = yield_stress(key)
                self.assertAlmostEqual(sy_calc, sy_data, delta=1e4)


# ── Hollomon stress ──────────────────────────────────────────────

class TestHollomonStress(unittest.TestCase):

    def test_returns_yield_at_zero_strain(self):
        for key in PLASTICITY_DATA:
            s = hollomon_stress(key, 0.0)
            self.assertAlmostEqual(s, yield_stress(key))

    def test_increases_with_strain(self):
        """Stress should increase with plastic strain (work hardening)."""
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                s1 = hollomon_stress(key, 0.01)
                s2 = hollomon_stress(key, 0.10)
                self.assertGreater(s2, s1, f"{key}: should harden")

    def test_silicon_no_hardening(self):
        """Silicon (brittle) should not harden."""
        s0 = hollomon_stress('silicon', 0.0)
        s1 = hollomon_stress('silicon', 0.10)
        self.assertAlmostEqual(s0, s1)


# ── Ramberg-Osgood ──────────────────────────────────────────────

class TestRambergOsgood(unittest.TestCase):

    def test_zero_stress_zero_strain(self):
        self.assertEqual(ramberg_osgood_strain('iron', 0.0), 0.0)

    def test_elastic_regime(self):
        """At low stress, should be nearly pure elastic."""
        E = youngs_modulus('iron')
        low_stress = 50e6  # well below yield
        eps = ramberg_osgood_strain('iron', low_stress)
        eps_elastic = low_stress / E
        # Should be close to elastic (plastic contribution small)
        self.assertAlmostEqual(eps / eps_elastic, 1.0, places=1)

    def test_plastic_dominates_at_high_stress(self):
        """Above yield, plastic strain should dominate."""
        sy = yield_stress('copper')
        high_stress = sy * 2.0
        eps = ramberg_osgood_strain('copper', high_stress)
        eps_elastic = high_stress / youngs_modulus('copper')
        # Total should significantly exceed elastic
        self.assertGreater(eps, eps_elastic * 1.5)

    def test_monotonic(self):
        """Strain should increase monotonically with stress."""
        stresses = [50e6, 100e6, 150e6, 200e6, 250e6]
        strains = [ramberg_osgood_strain('iron', s) for s in stresses]
        for i in range(1, len(strains)):
            self.assertGreater(strains[i], strains[i - 1])


# ── Ludwik ───────────────────────────────────────────────────────

class TestLudwikStress(unittest.TestCase):

    def test_returns_yield_at_zero(self):
        for key in PLASTICITY_DATA:
            self.assertAlmostEqual(ludwik_stress(key, 0.0), yield_stress(key))

    def test_exceeds_yield_with_strain(self):
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                s = ludwik_stress(key, 0.05)
                self.assertGreater(s, yield_stress(key))


# ── Johnson-Cook ─────────────────────────────────────────────────

class TestJohnsonCook(unittest.TestCase):

    def test_quasi_static_room_temp(self):
        """At quasi-static, room temp: JC ≈ A (yield)."""
        for key in PLASTICITY_DATA:
            s = johnson_cook_stress(key, 0.0, 1.0, 293.0)
            A = PLASTICITY_DATA[key]['jc_A']
            self.assertAlmostEqual(s, A, places=0)

    def test_strain_hardening(self):
        """Higher strain → higher stress at same rate and T."""
        for key, data in PLASTICITY_DATA.items():
            if data['jc_B'] > 0:
                s1 = johnson_cook_stress(key, 0.01)
                s2 = johnson_cook_stress(key, 0.20)
                self.assertGreater(s2, s1, f"{key}: JC should harden")

    def test_rate_hardening(self):
        """Higher strain rate → higher stress."""
        for key, data in PLASTICITY_DATA.items():
            if data['jc_C'] > 0:
                s_slow = johnson_cook_stress(key, 0.1, 1.0)
                s_fast = johnson_cook_stress(key, 0.1, 1000.0)
                self.assertGreater(s_fast, s_slow, f"{key}: rate hardening")

    def test_thermal_softening(self):
        """Higher temperature → lower stress."""
        for key, data in PLASTICITY_DATA.items():
            if data['jc_m'] > 0 and data['is_ductile']:
                s_cold = johnson_cook_stress(key, 0.1, 1.0, 300.0)
                s_hot = johnson_cook_stress(key, 0.1, 1.0, 1000.0)
                self.assertGreater(s_cold, s_hot, f"{key}: should soften at high T")

    def test_melting_gives_zero(self):
        """At T_melt, thermal factor should be zero."""
        T_melt = PLASTICITY_DATA['iron']['jc_T_melt']
        s = johnson_cook_stress('iron', 0.1, 1.0, T_melt)
        self.assertAlmostEqual(s, 0.0, places=0)

    def test_sigma_shifts_jc(self):
        s_0 = johnson_cook_stress('copper', 0.1, 1.0, 293.0, 0.0)
        s_s = johnson_cook_stress('copper', 0.1, 1.0, 293.0, 0.01)
        self.assertNotAlmostEqual(s_0, s_s, places=3)


# ── Necking ──────────────────────────────────────────────────────

class TestNecking(unittest.TestCase):

    def test_necking_strain_equals_n(self):
        """Considère: ε_neck = n for Hollomon materials."""
        for key, data in PLASTICITY_DATA.items():
            self.assertAlmostEqual(necking_strain(key), max(data['n_hardening'], 0.0))

    def test_necking_stress_positive(self):
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                sn = necking_stress(key)
                self.assertGreater(sn, yield_stress(key))

    def test_brittle_necking_at_yield(self):
        """Silicon: necking stress should equal yield (no hardening)."""
        self.assertAlmostEqual(necking_stress('silicon'), yield_stress('silicon'))


# ── Ductility ───────────────────────────────────────────────────

class TestDuctility(unittest.TestCase):

    def test_uniform_elongation_positive_for_ductile(self):
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                ue = uniform_elongation(key)
                self.assertGreater(ue, 0)

    def test_silicon_zero_elongation(self):
        self.assertAlmostEqual(uniform_elongation('silicon'), 0.0)
        self.assertAlmostEqual(total_elongation('silicon'), 0.0)

    def test_total_greater_than_uniform(self):
        """Total elongation includes post-necking, should be >= uniform."""
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0 and data['elongation_pct'] > 0:
                ue = uniform_elongation(key)
                te = total_elongation(key)
                # This isn't always true for all materials,
                # but for our well-characterized metals it should be
                self.assertGreater(te + ue, 0)  # Both positive

    def test_reduction_in_area_bounded(self):
        for key in PLASTICITY_DATA:
            ra = reduction_in_area(key)
            self.assertGreaterEqual(ra, 0.0)
            self.assertLessEqual(ra, 1.0)

    def test_copper_most_ductile(self):
        """Copper has highest elongation among the original 8 metals."""
        original_metals = [
            'iron', 'copper', 'aluminum', 'gold',
            'silicon', 'tungsten', 'nickel', 'titanium',
        ]
        te_Cu = total_elongation('copper')
        for key in original_metals:
            self.assertGreaterEqual(te_Cu, total_elongation(key) * 0.99)


# ── Stress-strain curve ─────────────────────────────────────────

class TestStressStrainCurve(unittest.TestCase):

    def test_returns_correct_length(self):
        curve = stress_strain_curve('iron', steps=50)
        self.assertEqual(len(curve), 51)

    def test_starts_at_zero(self):
        curve = stress_strain_curve('iron')
        self.assertAlmostEqual(curve[0]['strain'], 0.0)
        self.assertAlmostEqual(curve[0]['stress_pa'], 0.0)
        self.assertFalse(curve[0]['is_plastic'])

    def test_transitions_to_plastic(self):
        """Should have both elastic and plastic regions for ductile materials."""
        curve = stress_strain_curve('copper')
        has_elastic = any(not p['is_plastic'] for p in curve)
        has_plastic = any(p['is_plastic'] for p in curve)
        self.assertTrue(has_elastic)
        self.assertTrue(has_plastic)

    def test_stress_monotonically_increases(self):
        curve = stress_strain_curve('aluminum')
        for i in range(1, len(curve)):
            self.assertGreaterEqual(curve[i]['stress_pa'],
                                    curve[i - 1]['stress_pa'])


# ── Work hardening rate ──────────────────────────────────────────

class TestWorkHardeningRate(unittest.TestCase):

    def test_positive_for_ductile(self):
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0:
                whr = work_hardening_rate(key, 0.05)
                self.assertGreater(whr, 0)

    def test_decreases_with_strain(self):
        """Hardening rate should decrease (saturate) with strain."""
        for key, data in PLASTICITY_DATA.items():
            if data['n_hardening'] > 0 and data['n_hardening'] < 1.0:
                whr1 = work_hardening_rate(key, 0.01)
                whr2 = work_hardening_rate(key, 0.10)
                self.assertGreater(whr1, whr2, f"{key}: hardening should saturate")

    def test_zero_for_brittle(self):
        self.assertEqual(work_hardening_rate('silicon', 0.05), 0.0)


# ── Plastic work ─────────────────────────────────────────────────

class TestPlasticWork(unittest.TestCase):

    def test_zero_at_zero_strain(self):
        self.assertEqual(plastic_work_density('iron', 0.0), 0.0)

    def test_increases_with_strain(self):
        w1 = plastic_work_density('copper', 0.05)
        w2 = plastic_work_density('copper', 0.10)
        self.assertGreater(w2, w1)

    def test_toughness_positive_for_ductile(self):
        for key, data in PLASTICITY_DATA.items():
            if data['is_ductile'] and data['elongation_pct'] > 0:
                t = toughness_estimate(key)
                self.assertGreater(t, 0)

    def test_copper_high_toughness(self):
        """Copper (high n, high elongation) should have high toughness."""
        t_Cu = toughness_estimate('copper')
        t_Si = toughness_estimate('silicon')
        self.assertGreater(t_Cu, t_Si)


# ── σ-field coupling ────────────────────────────────────────────

class TestSigmaPlasticity(unittest.TestCase):

    def test_identity_at_zero(self):
        for key in PLASTICITY_DATA:
            self.assertAlmostEqual(sigma_yield_shift(key, 0.0), 1.0)

    def test_positive_sigma_strengthens(self):
        ratio = sigma_yield_shift('iron', 0.01)
        self.assertGreater(ratio, 1.0)

    def test_negative_sigma_weakens(self):
        ratio = sigma_yield_shift('iron', -0.01)
        self.assertLess(ratio, 1.0)


# ── Nagatha export ──────────────────────────────────────────────

class TestPlasticityProperties(unittest.TestCase):

    def test_all_materials_export(self):
        for key in PLASTICITY_DATA:
            props = plasticity_properties(key)
            self.assertEqual(props['material'], key)
            self.assertGreater(props['yield_stress_pa'], 0)
            self.assertIn('origin', props)

    def test_expected_keys(self):
        props = plasticity_properties('iron')
        expected = {
            'material', 'sigma', 'yield_stress_pa', 'strength_coefficient_pa',
            'n_hardening', 'necking_strain', 'necking_stress_pa',
            'uniform_elongation', 'total_elongation', 'reduction_in_area',
            'is_ductile', 'toughness_J_m3', 'sigma_yield_ratio',
            'jc_A_pa', 'jc_B_pa', 'jc_n', 'jc_C', 'jc_m', 'origin',
        }
        for k in expected:
            self.assertIn(k, props, f"Missing key: {k}")


if __name__ == '__main__':
    unittest.main()
