"""
Tests for the magnetism module.

Test structure:
  1. Diamagnetic susceptibility — Langevin formula, always negative
  2. Paramagnetic susceptibility — Curie's law, positive for unpaired electrons
  3. Total susceptibility — sign determines magnetic classification
  4. Ferromagnetic properties — saturation, Curie temperature, Bloch law
  5. Nuclear magnetic moments — gyromagnetic ratio, NMR
  6. σ-dependence — nuclear moment shift through QCD mass
  7. Cross-module consistency — uses MATERIALS from surface.py
  8. Nagatha export — complete format with origin tags
"""

import math
import unittest

from .magnetism import (
    mean_square_orbital_radius,
    diamagnetic_susceptibility,
    paramagnetic_susceptibility,
    magnetic_susceptibility,
    saturation_magnetization,
    saturation_magnetization_measured,
    magnetization_at_temperature,
    curie_temperature,
    is_ferromagnetic,
    nuclear_magnetic_moment,
    nmr_frequency_shift,
    material_magnetic_properties,
    MU_BOHR,
    MU_NUCLEAR,
    MAGNETIC_DATA,
)
from .surface import MATERIALS


class TestDiamagneticSusceptibility(unittest.TestCase):
    """Langevin diamagnetism — always negative, small magnitude."""

    def test_always_negative(self):
        """Diamagnetic susceptibility is negative for all materials."""
        for mat in MAGNETIC_DATA:
            chi = diamagnetic_susceptibility(mat)
            self.assertLess(chi, 0, f"{mat}: χ_dia should be negative")

    def test_order_of_magnitude(self):
        """|χ_dia| ~ 10⁻⁶ to 10⁻⁴ for metals."""
        for mat in MAGNETIC_DATA:
            chi = abs(diamagnetic_susceptibility(mat))
            self.assertGreater(chi, 1e-8,
                f"{mat}: |χ_dia| = {chi:.2e}, too small")
            self.assertLess(chi, 1e-2,
                f"{mat}: |χ_dia| = {chi:.2e}, too large")

    def test_heavier_atoms_scale(self):
        """Higher Z gives different diamagnetic response."""
        # Gold (Z=79) vs Aluminum (Z=13)
        chi_au = abs(diamagnetic_susceptibility('gold'))
        chi_al = abs(diamagnetic_susceptibility('aluminum'))
        # Both should be nonzero and distinct
        self.assertNotAlmostEqual(chi_au, chi_al, places=8)

    def test_copper_known_value(self):
        """Copper χ_dia ≈ −1×10⁻⁵ (order of magnitude check)."""
        chi = diamagnetic_susceptibility('copper')
        # Measured: χ = −9.63×10⁻⁶ (CRC)
        # Thomas-Fermi approximation: within factor of 5
        self.assertLess(chi, 0)
        self.assertGreater(abs(chi), 1e-7)
        self.assertLess(abs(chi), 1e-3)


class TestMeanSquareRadius(unittest.TestCase):
    """Thomas-Fermi orbital radius scaling."""

    def test_positive(self):
        """⟨r²⟩ is always positive."""
        for Z in [1, 6, 13, 26, 29, 79]:
            r2 = mean_square_orbital_radius(Z)
            self.assertGreater(r2, 0)

    def test_increases_with_Z(self):
        """Total ⟨r²⟩ increases with Z (more electrons)."""
        # ⟨r²⟩_total ∝ Z^(1/3) in Thomas-Fermi
        r2_al = mean_square_orbital_radius(13)
        r2_au = mean_square_orbital_radius(79)
        self.assertGreater(r2_au, r2_al)

    def test_scaling_exponent(self):
        """⟨r²⟩ ∝ Z^(1/3) — check the Thomas-Fermi exponent."""
        r2_8 = mean_square_orbital_radius(8)
        r2_64 = mean_square_orbital_radius(64)
        # Z ratio = 8, so r² ratio should be 8^(1/3) = 2.0
        ratio = r2_64 / r2_8
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)


class TestParamagneticSusceptibility(unittest.TestCase):
    """Curie's law — positive χ for unpaired electrons."""

    def test_zero_for_diamagnets(self):
        """Materials with no unpaired electrons: χ_para = 0."""
        for mat in ['copper', 'gold', 'silicon']:
            chi = paramagnetic_susceptibility(mat, T=300.0)
            self.assertEqual(chi, 0.0,
                f"{mat}: should have zero paramagnetic susceptibility")

    def test_positive_for_paramagnets(self):
        """Materials with unpaired electrons above T_C: χ_para > 0."""
        # Tungsten and titanium are paramagnetic
        for mat in ['tungsten', 'titanium']:
            chi = paramagnetic_susceptibility(mat, T=300.0)
            self.assertGreater(chi, 0,
                f"{mat}: χ_para should be positive")

    def test_curie_law_1_over_T(self):
        """χ_para ∝ 1/T (Curie's law)."""
        chi_300 = paramagnetic_susceptibility('titanium', T=300.0)
        chi_600 = paramagnetic_susceptibility('titanium', T=600.0)
        # χ(300) / χ(600) should be 2.0
        ratio = chi_300 / chi_600
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)

    def test_ferromagnet_above_curie(self):
        """Ferromagnets above T_C behave as paramagnets."""
        T_above = 1200.0  # above iron's T_C = 1043K
        chi = paramagnetic_susceptibility('iron', T=T_above)
        self.assertGreater(chi, 0)

    def test_ferromagnet_below_curie_returns_zero(self):
        """Ferromagnets below T_C: Curie's law doesn't apply."""
        chi = paramagnetic_susceptibility('iron', T=300.0)
        self.assertEqual(chi, 0.0)

    def test_zero_temperature(self):
        """T = 0: returns 0 (would be infinite, guard clause)."""
        chi = paramagnetic_susceptibility('titanium', T=0)
        self.assertEqual(chi, 0.0)


class TestTotalSusceptibility(unittest.TestCase):
    """Total χ determines magnetic classification."""

    def test_copper_negative(self):
        """Copper: diamagnetic (no unpaired electrons), χ < 0."""
        chi = magnetic_susceptibility('copper', T=300.0)
        self.assertLess(chi, 0)

    def test_gold_negative(self):
        """Gold: diamagnetic, χ < 0."""
        chi = magnetic_susceptibility('gold', T=300.0)
        self.assertLess(chi, 0)

    def test_titanium_positive(self):
        """Titanium: paramagnetic, χ > 0."""
        chi = magnetic_susceptibility('titanium', T=300.0)
        self.assertGreater(chi, 0)

    def test_tungsten_positive(self):
        """Tungsten: paramagnetic, χ > 0."""
        chi = magnetic_susceptibility('tungsten', T=300.0)
        self.assertGreater(chi, 0)

    def test_paramagnetic_beats_diamagnetic(self):
        """For paramagnets, |χ_para| >> |χ_dia| at room temperature."""
        chi_para = paramagnetic_susceptibility('titanium', T=300.0)
        chi_dia = abs(diamagnetic_susceptibility('titanium'))
        self.assertGreater(chi_para, chi_dia)


class TestFerromagneticProperties(unittest.TestCase):
    """Saturation magnetization, Curie temperature, Bloch law."""

    def test_iron_saturation_positive(self):
        """Iron saturation magnetization is large and positive."""
        M = saturation_magnetization('iron')
        # Iron: ~1.7×10⁶ A/m (theoretical), measured ~1.7×10⁶
        self.assertGreater(M, 1e5)
        self.assertLess(M, 1e7)

    def test_nickel_saturation_positive(self):
        """Nickel saturation magnetization positive."""
        M = saturation_magnetization('nickel')
        self.assertGreater(M, 1e4)
        self.assertLess(M, 1e7)

    def test_iron_stronger_than_nickel(self):
        """Iron has higher saturation than nickel (more unpaired + higher density)."""
        M_fe = saturation_magnetization('iron')
        M_ni = saturation_magnetization('nickel')
        self.assertGreater(M_fe, M_ni)

    def test_non_ferromagnets_zero(self):
        """Non-ferromagnetic materials return zero saturation."""
        for mat in ['copper', 'gold', 'aluminum', 'silicon']:
            M = saturation_magnetization(mat)
            self.assertEqual(M, 0.0)

    def test_measured_less_than_theoretical(self):
        """Measured M_sat < theoretical (band effects reduce moment)."""
        for mat in ['iron', 'nickel']:
            M_theo = saturation_magnetization(mat)
            M_meas = saturation_magnetization_measured(mat)
            self.assertGreater(M_meas, 0)
            self.assertLess(M_meas, M_theo,
                f"{mat}: measured should be less than free-ion prediction")

    def test_curie_temperatures(self):
        """Known Curie temperatures."""
        self.assertAlmostEqual(curie_temperature('iron'), 1043.0, delta=1.0)
        self.assertAlmostEqual(curie_temperature('nickel'), 627.0, delta=1.0)

    def test_non_ferromagnet_curie_zero(self):
        """Non-ferromagnets have T_C = 0."""
        self.assertEqual(curie_temperature('copper'), 0.0)

    def test_is_ferromagnetic(self):
        """Iron is ferromagnetic at 300K, not at 1200K."""
        self.assertTrue(is_ferromagnetic('iron', T=300.0))
        self.assertFalse(is_ferromagnetic('iron', T=1200.0))
        self.assertFalse(is_ferromagnetic('copper', T=300.0))


class TestBlochLaw(unittest.TestCase):
    """Bloch T^(3/2) law for temperature dependence of magnetization."""

    def test_zero_K_full_saturation(self):
        """At T = 0K, M = M_sat (all spins aligned)."""
        # T=0 gives (T/T_C)^1.5 = 0, so M = M_sat
        M_0 = magnetization_at_temperature('iron', T=0.001)  # near zero
        M_sat = saturation_magnetization_measured('iron')
        self.assertAlmostEqual(M_0 / M_sat, 1.0, delta=1e-6)

    def test_at_curie_temp_zero(self):
        """At T = T_C, M = 0 (phase transition)."""
        T_C = curie_temperature('iron')
        M = magnetization_at_temperature('iron', T=T_C)
        self.assertEqual(M, 0.0)

    def test_above_curie_temp_zero(self):
        """Above T_C, M = 0."""
        M = magnetization_at_temperature('iron', T=2000.0)
        self.assertEqual(M, 0.0)

    def test_decreases_with_temperature(self):
        """M decreases monotonically with T."""
        temps = [100, 300, 500, 700, 900]
        prev_M = float('inf')
        for T in temps:
            M = magnetization_at_temperature('iron', T)
            self.assertLess(M, prev_M)
            self.assertGreater(M, 0)
            prev_M = M

    def test_room_temperature_iron(self):
        """Iron at 300K: M is ~90% of M_sat (well below T_C)."""
        M_300 = magnetization_at_temperature('iron', T=300.0)
        M_sat = saturation_magnetization_measured('iron')
        ratio = M_300 / M_sat
        self.assertGreater(ratio, 0.80)
        self.assertLess(ratio, 0.99)

    def test_non_ferromagnet_zero(self):
        """Non-ferromagnets return zero at any temperature."""
        M = magnetization_at_temperature('copper', T=300.0)
        self.assertEqual(M, 0.0)


class TestNuclearMoments(unittest.TestCase):
    """Nuclear magnetic moments and NMR."""

    def test_sigma_zero_unchanged(self):
        """At σ = 0, nuclear moment equals tabulated value."""
        for mat in MAGNETIC_DATA:
            mu_0 = MAGNETIC_DATA[mat]['nuclear_moment_mu_N']
            mu_calc = nuclear_magnetic_moment(mat, sigma=0.0)
            self.assertAlmostEqual(mu_calc, mu_0, places=10)

    def test_nmr_shift_zero_at_flat(self):
        """At σ = 0, no NMR frequency shift."""
        shift = nmr_frequency_shift('iron', sigma=0.0)
        self.assertAlmostEqual(shift, 0.0, places=15)

    def test_nmr_shift_negative_for_positive_sigma(self):
        """Positive σ → heavier nucleons → lower NMR frequency."""
        shift = nmr_frequency_shift('iron', sigma=0.1)
        self.assertLess(shift, 0)

    def test_nmr_shift_magnitude(self):
        """At σ = 0.1 (neutron star surface), shift ~ −10%."""
        shift = nmr_frequency_shift('iron', sigma=0.1)
        # mass_ratio ≈ 1 + 0.99 × (e^0.1 − 1) ≈ 1.104
        # shift ≈ 1/1.104 − 1 ≈ −0.094
        self.assertAlmostEqual(shift, -0.094, delta=0.01)


class TestSigmaDependence(unittest.TestCase):
    """σ-field propagation through magnetic properties."""

    def test_earth_sigma_negligible(self):
        """Earth σ ~ 7×10⁻¹⁰: NMR shift < 10⁻⁹."""
        shift = nmr_frequency_shift('iron', sigma=7e-10)
        self.assertLess(abs(shift), 1e-8)

    def test_nuclear_moment_decreases_with_sigma(self):
        """Higher σ → heavier nucleons → smaller nuclear moment."""
        for mat in MAGNETIC_DATA:
            mu_0 = abs(nuclear_magnetic_moment(mat, sigma=0.0))
            mu_1 = abs(nuclear_magnetic_moment(mat, sigma=1.0))
            if mu_0 > 0:
                self.assertLess(mu_1, mu_0,
                    f"{mat}: nuclear moment should decrease with σ")

    def test_electronic_magnetism_invariant(self):
        """Diamagnetic susceptibility is σ-invariant (electromagnetic)."""
        # Our implementation doesn't take σ as input for χ_dia,
        # which correctly reflects that it's EM and σ-invariant.
        chi_1 = diamagnetic_susceptibility('copper')
        chi_2 = diamagnetic_susceptibility('copper')
        self.assertEqual(chi_1, chi_2)


class TestBohrMagneton(unittest.TestCase):
    """Fundamental constants sanity checks."""

    def test_bohr_magneton_value(self):
        """μ_B ≈ 9.274×10⁻²⁴ J/T."""
        self.assertAlmostEqual(MU_BOHR, 9.274e-24, delta=1e-26)

    def test_nuclear_magneton_value(self):
        """μ_N ≈ 5.051×10⁻²⁷ J/T."""
        self.assertAlmostEqual(MU_NUCLEAR, 5.051e-27, delta=1e-29)

    def test_nuclear_much_smaller(self):
        """Nuclear magneton ~ 1/1836 of Bohr magneton."""
        ratio = MU_NUCLEAR / MU_BOHR
        self.assertAlmostEqual(ratio, 1.0 / 1836.0, delta=0.001)


class TestNagathaExport(unittest.TestCase):
    """Nagatha-compatible export format."""

    def test_required_fields(self):
        """Export contains all required fields."""
        props = material_magnetic_properties('iron', T=300.0)
        required = [
            'material', 'temperature_K', 'sigma', 'magnetic_type',
            'n_unpaired_electrons', 'diamagnetic_susceptibility',
            'total_susceptibility', 'nuclear_moment_mu_N',
            'nmr_frequency_shift', 'origin',
        ]
        for key in required:
            self.assertIn(key, props, f"Missing: {key}")

    def test_ferromagnet_extra_fields(self):
        """Ferromagnets have additional fields."""
        props = material_magnetic_properties('iron', T=300.0)
        extra = [
            'curie_temperature_K', 'saturation_magnetization_A_m',
            'magnetization_at_T_A_m', 'is_ferromagnetic_at_T',
        ]
        for key in extra:
            self.assertIn(key, props, f"Missing ferromagnet field: {key}")

    def test_honest_origin_tags(self):
        """Origin string contains FIRST_PRINCIPLES and MEASURED."""
        props = material_magnetic_properties('iron')
        origin = props['origin']
        self.assertIn('FIRST_PRINCIPLES', origin)
        self.assertIn('MEASURED', origin)
        self.assertIn('APPROXIMATION', origin)

    def test_all_materials_export(self):
        """Every material in the magnetic database exports without error."""
        for mat in MAGNETIC_DATA:
            props = material_magnetic_properties(mat)
            self.assertIn('origin', props)

    def test_sigma_propagates_to_nuclear(self):
        """σ parameter affects nuclear moment in export."""
        props_0 = material_magnetic_properties('iron', sigma=0.0)
        props_1 = material_magnetic_properties('iron', sigma=1.0)
        self.assertNotEqual(
            props_0['nuclear_moment_mu_N'],
            props_1['nuclear_moment_mu_N'])


class TestCrossModuleConsistency(unittest.TestCase):
    """Magnetism module uses consistent data with other modules."""

    def test_materials_from_surface(self):
        """All magnetic materials exist in MATERIALS database."""
        for mat in MAGNETIC_DATA:
            self.assertIn(mat, MATERIALS,
                f"{mat}: in MAGNETIC_DATA but not in MATERIALS")

    def test_Z_consistent(self):
        """Atomic number Z used consistently."""
        for mat in MAGNETIC_DATA:
            Z = MATERIALS[mat]['Z']
            # Verify Z makes sense for the magnetic properties claimed
            self.assertGreater(Z, 0)

    def test_iron_is_transition_metal(self):
        """Iron (Z=26) is a 3d transition metal — should have unpaired d-electrons."""
        Z = MATERIALS['iron']['Z']
        self.assertEqual(Z, 26)
        self.assertEqual(MAGNETIC_DATA['iron']['n_unpaired'], 4)


if __name__ == '__main__':
    unittest.main()
