"""
Tests for the elasticity module.

Test structure:
  1. Lamé parameters — relationships to E, G, ν
  2. Stress-strain — Hooke's law in all loading modes
  3. Poisson effect — transverse strain and volume change
  4. Strain energy density — work integral
  5. Von Mises yield criterion — distortional energy
  6. Moduli relationships — Lamé ↔ engineering constants
  7. σ-dependence — elastic shift through nuclear mass
  8. Nagatha export — complete format with origin tags
"""

import math
import unittest

from .elasticity import (
    lame_lambda, lame_mu,
    uniaxial_stress, shear_stress, hydrostatic_stress,
    transverse_strain, volume_change_uniaxial,
    strain_energy_density_uniaxial, strain_energy_density_shear,
    strain_energy_density_hydrostatic,
    von_mises_stress, is_yielded,
    moduli_from_lame, p_wave_modulus,
    sigma_elastic_shift,
    material_elastic_properties,
)
from .mechanical import (
    bulk_modulus, youngs_modulus, shear_modulus, MECHANICAL_DATA,
)


class TestLameParameters(unittest.TestCase):
    """Lamé parameters — isotropic elasticity identities."""

    def test_mu_equals_shear(self):
        """μ = G (second Lamé parameter is shear modulus)."""
        for mat in MECHANICAL_DATA:
            mu = lame_mu(mat)
            G = shear_modulus(mat)
            self.assertAlmostEqual(mu, G, places=2)

    def test_lambda_positive(self):
        """λ > 0 for all materials with ν > 0."""
        for mat in MECHANICAL_DATA:
            lam = lame_lambda(mat)
            self.assertGreater(lam, 0, f"{mat}: λ should be positive")

    def test_bulk_from_lame(self):
        """K = λ + 2μ/3 (exact identity)."""
        for mat in MECHANICAL_DATA:
            lam = lame_lambda(mat)
            mu = lame_mu(mat)
            K_from_lame = lam + 2.0 * mu / 3.0
            K_direct = bulk_modulus(mat)
            self.assertAlmostEqual(K_from_lame / K_direct, 1.0, places=6,
                msg=f"{mat}: K from Lamé should match K direct")

    def test_iron_order_of_magnitude(self):
        """Iron: λ ~ 100 GPa (order of magnitude check)."""
        lam = lame_lambda('iron')
        self.assertGreater(lam, 50e9)
        self.assertLess(lam, 300e9)


class TestStressStrain(unittest.TestCase):
    """Hooke's law — stress from strain in all modes."""

    def test_uniaxial_positive(self):
        """Tensile strain → positive stress."""
        sigma = uniaxial_stress('iron', 0.001)  # 0.1% strain
        self.assertGreater(sigma, 0)

    def test_uniaxial_iron_order(self):
        """Iron at 0.1% strain: ~200 MPa (within elastic range)."""
        sigma = uniaxial_stress('iron', 0.001)
        self.assertGreater(sigma, 100e6)
        self.assertLess(sigma, 500e6)

    def test_uniaxial_linearity(self):
        """Doubling strain doubles stress (Hooke's law)."""
        s1 = uniaxial_stress('copper', 0.001)
        s2 = uniaxial_stress('copper', 0.002)
        self.assertAlmostEqual(s2 / s1, 2.0, places=6)

    def test_shear_stress_positive(self):
        """Positive shear strain → positive shear stress."""
        tau = shear_stress('iron', 0.001)
        self.assertGreater(tau, 0)

    def test_hydrostatic_compression(self):
        """Compression (negative ΔV/V) → positive pressure."""
        P = hydrostatic_stress('iron', -0.001)
        self.assertGreater(P, 0)

    def test_hydrostatic_expansion(self):
        """Expansion (positive ΔV/V) → negative pressure (tension)."""
        P = hydrostatic_stress('iron', 0.001)
        self.assertLess(P, 0)


class TestPoissonEffect(unittest.TestCase):
    """Transverse strain and volume change."""

    def test_transverse_opposite_sign(self):
        """Transverse strain has opposite sign to axial."""
        et = transverse_strain('iron', 0.01)
        self.assertLess(et, 0, "Tension should cause lateral contraction")

    def test_transverse_magnitude(self):
        """ε_transverse = -ν × ε_axial."""
        nu = MECHANICAL_DATA['copper']['poisson_ratio']
        et = transverse_strain('copper', 0.01)
        self.assertAlmostEqual(et, -nu * 0.01, places=10)

    def test_volume_change_positive(self):
        """Tension (ε>0) with ν<0.5 gives positive ΔV/V."""
        dV = volume_change_uniaxial('iron', 0.01)
        self.assertGreater(dV, 0)

    def test_volume_change_formula(self):
        """ΔV/V = (1 - 2ν)ε."""
        nu = MECHANICAL_DATA['aluminum']['poisson_ratio']
        dV = volume_change_uniaxial('aluminum', 0.01)
        expected = (1.0 - 2.0 * nu) * 0.01
        self.assertAlmostEqual(dV, expected, places=10)


class TestStrainEnergy(unittest.TestCase):
    """Elastic strain energy density — work integral."""

    def test_uniaxial_positive(self):
        """Strain energy is always positive."""
        u = strain_energy_density_uniaxial('iron', 0.001)
        self.assertGreater(u, 0)

    def test_uniaxial_quadratic(self):
        """Energy scales as ε² (Hooke's law)."""
        u1 = strain_energy_density_uniaxial('iron', 0.001)
        u2 = strain_energy_density_uniaxial('iron', 0.002)
        self.assertAlmostEqual(u2 / u1, 4.0, places=6)

    def test_shear_energy_positive(self):
        """Shear strain energy is positive."""
        u = strain_energy_density_shear('iron', 0.001)
        self.assertGreater(u, 0)

    def test_hydrostatic_energy_positive(self):
        """Hydrostatic strain energy is positive."""
        u = strain_energy_density_hydrostatic('iron', -0.001)
        self.assertGreater(u, 0)

    def test_uniaxial_formula(self):
        """u = ½Eε²."""
        E = youngs_modulus('copper')
        eps = 0.001
        u = strain_energy_density_uniaxial('copper', eps)
        self.assertAlmostEqual(u, 0.5 * E * eps ** 2, places=2)


class TestVonMises(unittest.TestCase):
    """Von Mises yield criterion."""

    def test_uniaxial(self):
        """Uniaxial tension: σ_vm = |σ₁|."""
        sigma_vm = von_mises_stress(100e6, 0, 0)
        self.assertAlmostEqual(sigma_vm, 100e6, places=0)

    def test_pure_shear(self):
        """Pure shear: σ₁ = τ, σ₂ = -τ, σ₃ = 0 → σ_vm = τ√3."""
        tau = 50e6
        sigma_vm = von_mises_stress(tau, -tau, 0)
        self.assertAlmostEqual(sigma_vm, tau * math.sqrt(3), places=0)

    def test_hydrostatic_zero(self):
        """Hydrostatic stress: σ_vm = 0 (no distortion)."""
        P = 100e6
        sigma_vm = von_mises_stress(P, P, P)
        self.assertAlmostEqual(sigma_vm, 0.0, places=2)

    def test_yielding(self):
        """is_yielded returns True when exceeded."""
        self.assertTrue(is_yielded(300e6, 0, 0, 200e6))
        self.assertFalse(is_yielded(100e6, 0, 0, 200e6))


class TestModuliRelationships(unittest.TestCase):
    """Consistency of elastic moduli conversions."""

    def test_round_trip_lame(self):
        """(E, ν) → (λ, μ) → (E, ν) round trip."""
        for mat in MECHANICAL_DATA:
            lam = lame_lambda(mat)
            mu = lame_mu(mat)
            result = moduli_from_lame(lam, mu)
            E = youngs_modulus(mat)
            nu = MECHANICAL_DATA[mat]['poisson_ratio']
            self.assertAlmostEqual(result['E_pa'] / E, 1.0, places=5,
                msg=f"{mat}: E round-trip")
            self.assertAlmostEqual(result['poisson_ratio'], nu, places=5,
                msg=f"{mat}: ν round-trip")

    def test_p_wave_modulus(self):
        """M = K + 4G/3."""
        for mat in MECHANICAL_DATA:
            M = p_wave_modulus(mat)
            K = bulk_modulus(mat)
            G = shear_modulus(mat)
            self.assertAlmostEqual(M, K + 4 * G / 3, places=2)

    def test_p_wave_greater_than_bulk(self):
        """M > K always (since G > 0)."""
        for mat in MECHANICAL_DATA:
            M = p_wave_modulus(mat)
            K = bulk_modulus(mat)
            self.assertGreater(M, K)


class TestSigmaDependence(unittest.TestCase):
    """σ-field shifts elastic properties."""

    def test_shift_unity_at_zero(self):
        """E(σ=0)/E(0) = 1."""
        for mat in MECHANICAL_DATA:
            ratio = sigma_elastic_shift(mat, 0.0)
            self.assertAlmostEqual(ratio, 1.0, places=10)

    def test_shift_increases_with_sigma(self):
        """E(σ) > E(0) for σ > 0 (stiffer lattice)."""
        for mat in MECHANICAL_DATA:
            ratio = sigma_elastic_shift(mat, 0.1)
            self.assertGreater(ratio, 1.0,
                f"{mat}: elastic modulus should increase with σ")

    def test_lame_shifts_with_sigma(self):
        """Lamé parameters shift under σ."""
        for mat in MECHANICAL_DATA:
            lam_0 = lame_lambda(mat, 0.0)
            lam_s = lame_lambda(mat, 0.1)
            self.assertNotAlmostEqual(lam_0, lam_s, places=2)

    def test_earth_sigma_negligible(self):
        """At Earth σ ~ 7×10⁻¹⁰: shift < 10⁻⁸."""
        sigma_earth = 7e-10
        ratio = sigma_elastic_shift('iron', sigma_earth)
        self.assertAlmostEqual(ratio, 1.0, places=8)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export."""

    def test_all_materials_export(self):
        """All materials produce valid export dicts."""
        for mat in MECHANICAL_DATA:
            props = material_elastic_properties(mat)
            self.assertIn('youngs_modulus_pa', props)
            self.assertIn('bulk_modulus_pa', props)
            self.assertIn('lame_lambda_pa', props)
            self.assertIn('p_wave_modulus_pa', props)
            self.assertIn('origin_tag', props)

    def test_sigma_propagates(self):
        """σ value appears in export."""
        props = material_elastic_properties('iron', sigma=0.1)
        self.assertEqual(props['sigma'], 0.1)
        self.assertGreater(props['elastic_shift_ratio'], 1.0)

    def test_honest_origin_tags(self):
        """Origin tag includes FIRST_PRINCIPLES and MEASURED."""
        props = material_elastic_properties('iron')
        self.assertIn('FIRST_PRINCIPLES', props['origin_tag'])
        self.assertIn('MEASURED', props['origin_tag'])


if __name__ == '__main__':
    unittest.main()
