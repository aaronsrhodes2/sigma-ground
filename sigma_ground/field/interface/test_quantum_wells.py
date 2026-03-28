"""
Tests for quantum_wells.py — confinement, quantum dots, density of states.

Strategy:
  - Test particle-in-box E₁ scales as 1/L² and n²
  - Test 1 nm electron box ≈ 0.38 eV (textbook)
  - Test 3D cubic box degeneracy
  - Test finite well always has ≥ 1 bound state
  - Test finite well E < V₀
  - Test Brus equation: smaller dots → larger gap
  - Test CdSe QD: R=2nm → ~530nm green (MEASURED)
  - Test DOS: 3D ∝ √E, 2D = constant, 1D ∝ 1/√E
  - Test quantum well subbands scale as n²
  - Test critical radius (exciton Bohr radius)
  - Test Rule 9: full_report covers all fields

Reference values (MEASURED):
  CdSe QD R=2nm emission: ~520-540 nm (green)
  CdSe bulk gap: 1.74 eV (712 nm)
  Electron in 1nm box: ~0.376 eV
  3D DOS prefactor: ~1.06e56 states/(J·m³) for electrons
"""

import math
import unittest

from sigma_ground.field.interface.quantum_wells import (
    box_energy_1d_eV,
    box_energy_3d_eV,
    box_ground_state_eV,
    box_transition_wavelength_nm,
    degeneracy_3d_cubic,
    finite_well_bound_states,
    finite_well_ground_state_eV,
    tunneling_depth_m,
    brus_energy_eV,
    qd_emission_wavelength_nm,
    qd_radius_for_wavelength_nm,
    qd_color_rgb,
    confinement_energy_eV,
    dos_3d,
    dos_2d,
    dos_1d,
    dos_0d,
    quantum_well_subbands_eV,
    quantum_wire_subbands_eV,
    critical_radius_nm,
    size_vs_gap,
    quantum_wells_report,
    full_report,
    _QD_MATERIALS,
)


class TestParticleInBox1D(unittest.TestCase):
    """1D infinite square well."""

    def test_ground_state_1nm(self):
        """Electron in 1 nm box: E₁ ≈ 0.376 eV (textbook)."""
        E = box_ground_state_eV(1e-9)
        self.assertAlmostEqual(E, 0.376, delta=0.02)

    def test_energy_scales_n_squared(self):
        """Eₙ ∝ n²."""
        L = 2e-9
        E1 = box_energy_1d_eV(1, L)
        E2 = box_energy_1d_eV(2, L)
        E3 = box_energy_1d_eV(3, L)
        self.assertAlmostEqual(E2 / E1, 4.0, delta=0.001)
        self.assertAlmostEqual(E3 / E1, 9.0, delta=0.001)

    def test_energy_scales_inverse_L_squared(self):
        """E₁ ∝ 1/L²."""
        E_1nm = box_ground_state_eV(1e-9)
        E_2nm = box_ground_state_eV(2e-9)
        self.assertAlmostEqual(E_1nm / E_2nm, 4.0, delta=0.001)

    def test_heavier_particle_lower_energy(self):
        """Heavier particle → lower energy (E ∝ 1/m)."""
        from sigma_ground.field.constants import M_ELECTRON_KG, AMU_KG
        E_e = box_ground_state_eV(1e-9, M_ELECTRON_KG)
        E_p = box_ground_state_eV(1e-9, AMU_KG)
        self.assertGreater(E_e, E_p)

    def test_transition_wavelength(self):
        """2→1 transition gives a wavelength."""
        lam = box_transition_wavelength_nm(2, 1, 1e-9)
        self.assertGreater(lam, 0)
        self.assertTrue(math.isfinite(lam))


class TestParticleInBox3D(unittest.TestCase):
    """3D rectangular box."""

    def test_cubic_ground_state(self):
        """3D cubic box: E(1,1,1) = 3 × E₁(1D)."""
        L = 2e-9
        E_1d = box_energy_1d_eV(1, L)
        E_3d = box_energy_3d_eV(1, 1, 1, L)
        self.assertAlmostEqual(E_3d, 3 * E_1d, delta=0.001)

    def test_degeneracy_6(self):
        """n²=6 → (1,1,2) permutations → 3-fold degenerate."""
        self.assertEqual(degeneracy_3d_cubic(6), 3)

    def test_degeneracy_3(self):
        """n²=3 → (1,1,1) → non-degenerate."""
        self.assertEqual(degeneracy_3d_cubic(3), 1)

    def test_degeneracy_9(self):
        """n²=9 → (1,2,2) permutations + (2,2,1) + (2,1,2) → 3-fold."""
        self.assertEqual(degeneracy_3d_cubic(9), 3)

    def test_rectangular_removes_degeneracy(self):
        """Non-cubic box: E(1,1,2) ≠ E(1,2,1) if L₁ ≠ L₂."""
        E_112 = box_energy_3d_eV(1, 1, 2, 1e-9, 2e-9, 3e-9)
        E_121 = box_energy_3d_eV(1, 2, 1, 1e-9, 2e-9, 3e-9)
        self.assertNotAlmostEqual(E_112, E_121)


class TestFiniteWell(unittest.TestCase):
    """Finite square well."""

    def test_always_one_bound_state(self):
        """1D finite well always has at least 1 bound state."""
        n = finite_well_bound_states(0.01, 1e-9)  # very shallow
        self.assertGreaterEqual(n, 1)

    def test_more_states_deeper_well(self):
        """Deeper well → more bound states."""
        n_shallow = finite_well_bound_states(1.0, 5e-9)
        n_deep = finite_well_bound_states(10.0, 5e-9)
        self.assertGreaterEqual(n_deep, n_shallow)

    def test_more_states_wider_well(self):
        """Wider well → more bound states."""
        n_narrow = finite_well_bound_states(5.0, 1e-9)
        n_wide = finite_well_bound_states(5.0, 10e-9)
        self.assertGreaterEqual(n_wide, n_narrow)

    def test_ground_state_below_V0(self):
        """Ground state energy < V₀ (otherwise not bound)."""
        V0 = 2.0
        E = finite_well_ground_state_eV(V0, 5e-9)
        self.assertGreater(E, 0)
        self.assertLess(E, V0)

    def test_ground_state_below_infinite_well(self):
        """Finite well E₁ < infinite well E₁ (wavefunction leaks out)."""
        L = 5e-9
        V0 = 10.0
        E_finite = finite_well_ground_state_eV(V0, L)
        E_infinite = box_ground_state_eV(L)
        self.assertLess(E_finite, E_infinite)

    def test_tunneling_depth_positive(self):
        """Evanescent depth is positive."""
        d = tunneling_depth_m(5.0, 2.0)
        self.assertGreater(d, 0)

    def test_tunneling_depth_decreases_with_barrier(self):
        """Higher barrier → shorter evanescent tail."""
        d_low = tunneling_depth_m(2.0, 1.0)
        d_high = tunneling_depth_m(10.0, 1.0)
        self.assertGreater(d_low, d_high)


class TestQuantumDots(unittest.TestCase):
    """Brus equation and quantum dot properties."""

    def test_brus_larger_than_bulk(self):
        """QD gap > bulk gap (confinement widens gap)."""
        R = 2e-9
        Eg_qd = brus_energy_eV(R, 'CdSe')
        Eg_bulk = _QD_MATERIALS['CdSe'][3]
        self.assertGreater(Eg_qd, Eg_bulk)

    def test_brus_approaches_bulk(self):
        """Large QD → gap approaches bulk value."""
        R_large = 50e-9
        Eg = brus_energy_eV(R_large, 'CdSe')
        Eg_bulk = _QD_MATERIALS['CdSe'][3]
        self.assertAlmostEqual(Eg, Eg_bulk, delta=0.1)

    def test_smaller_dot_larger_gap(self):
        """Smaller QD → wider gap."""
        Eg_small = brus_energy_eV(1.5e-9, 'CdSe')
        Eg_large = brus_energy_eV(4e-9, 'CdSe')
        self.assertGreater(Eg_small, Eg_large)

    def test_CdSe_2nm_green(self):
        """CdSe R=2nm emits green (~500-560 nm) (MEASURED)."""
        lam = qd_emission_wavelength_nm(2e-9, 'CdSe')
        self.assertGreater(lam, 450)
        self.assertLess(lam, 600)

    def test_CdSe_1nm_blue(self):
        """CdSe R=1nm emits blue (<500 nm)."""
        lam = qd_emission_wavelength_nm(1e-9, 'CdSe')
        self.assertLess(lam, 500)

    def test_radius_for_wavelength_inverse(self):
        """qd_radius_for_wavelength is inverse of qd_emission_wavelength."""
        target = 550.0
        R = qd_radius_for_wavelength_nm(target, 'CdSe')
        lam_back = qd_emission_wavelength_nm(R, 'CdSe')
        self.assertAlmostEqual(lam_back, target, delta=1.0)

    def test_all_materials_work(self):
        """Brus equation works for all QD materials."""
        for key in _QD_MATERIALS:
            with self.subTest(material=key):
                Eg = brus_energy_eV(3e-9, key)
                self.assertGreater(Eg, 0)

    def test_color_visible_CdSe(self):
        """CdSe QD at 2nm has non-black RGB."""
        r, g, b = qd_color_rgb(2e-9, 'CdSe')
        self.assertTrue(r > 0 or g > 0 or b > 0)

    def test_confinement_energy(self):
        """Confinement energy is positive and scales as 1/R²."""
        E1 = confinement_energy_eV(1e-9)
        E2 = confinement_energy_eV(2e-9)
        self.assertGreater(E1, 0)
        self.assertAlmostEqual(E1 / E2, 4.0, delta=0.01)

    def test_critical_radius_positive(self):
        """Critical radius is positive and finite."""
        R_c = critical_radius_nm('CdSe')
        self.assertGreater(R_c, 0)
        self.assertLess(R_c, 100)


class TestDensityOfStates(unittest.TestCase):
    """Density of states in different dimensions."""

    def test_3d_sqrt_E(self):
        """3D DOS ∝ √E."""
        g1 = dos_3d(1.0)
        g4 = dos_3d(4.0)
        self.assertAlmostEqual(g4 / g1, 2.0, delta=0.01)

    def test_3d_zero_at_zero(self):
        """3D DOS = 0 at E = 0."""
        self.assertEqual(dos_3d(0.0), 0.0)

    def test_3d_negative_returns_zero(self):
        """3D DOS = 0 for E < 0."""
        self.assertEqual(dos_3d(-1.0), 0.0)

    def test_2d_constant(self):
        """2D DOS is a constant (no energy dependence)."""
        g = dos_2d()
        self.assertGreater(g, 0)

    def test_1d_diverges_at_edge(self):
        """1D DOS → ∞ at sub-band edge (van Hove singularity)."""
        g_near = dos_1d(0.001, 0.0)
        g_far = dos_1d(1.0, 0.0)
        self.assertGreater(g_near, g_far)

    def test_1d_zero_below_subband(self):
        """1D DOS = 0 below sub-band edge."""
        self.assertEqual(dos_1d(0.5, 1.0), 0.0)

    def test_0d_peaks_at_levels(self):
        """0D DOS peaks at discrete levels."""
        levels = [1.0, 2.0, 3.0]
        g_at_level = dos_0d(1.0, levels, 0.01)
        g_between = dos_0d(1.5, levels, 0.01)
        self.assertGreater(g_at_level, g_between)

    def test_0d_broadening(self):
        """Narrower broadening → sharper peaks."""
        levels = [1.0]
        g_broad = dos_0d(1.0, levels, 0.1)
        g_narrow = dos_0d(1.0, levels, 0.01)
        self.assertGreater(g_narrow, g_broad)


class TestQuantumWellSubbands(unittest.TestCase):
    """Quantum well and wire subbands."""

    def test_well_subbands_scale_n_squared(self):
        """Well subbands: Eₙ ∝ n²."""
        levels = quantum_well_subbands_eV(5e-9, n_max=3)
        E1 = levels[0][1]
        E2 = levels[1][1]
        E3 = levels[2][1]
        self.assertAlmostEqual(E2 / E1, 4.0, delta=0.01)
        self.assertAlmostEqual(E3 / E1, 9.0, delta=0.01)

    def test_wire_subbands_ordered(self):
        """Wire subbands are sorted by energy."""
        levels = quantum_wire_subbands_eV(3e-9, 3e-9, n_max=3)
        for i in range(len(levels) - 1):
            self.assertLessEqual(levels[i][2], levels[i+1][2])

    def test_size_vs_gap_curve(self):
        """size_vs_gap returns sensible curve."""
        curve = size_vs_gap('CdSe', 1.0, 10.0, 5)
        self.assertEqual(len(curve), 5)
        # Larger R → smaller Eg
        self.assertGreater(curve[0][1], curve[-1][1])


class TestReports(unittest.TestCase):
    """Rule 9: full_report."""

    def test_report_complete(self):
        """quantum_wells_report has required fields."""
        r = quantum_wells_report()
        required = ['well_width_nm', 'box_ground_state_eV',
                     'well_subbands_eV', 'brus_gap_eV',
                     'dot_emission_nm', 'critical_radius_nm', 'bulk_gap_eV']
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report(self):
        """full_report returns dict with extra fields."""
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertIn('finite_well_bound_states', r)
        self.assertIn('available_materials', r)
        self.assertGreater(len(r['available_materials']), 5)


if __name__ == '__main__':
    unittest.main()
