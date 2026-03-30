"""
Tests for MaterialSample — Golden Rule 10 volume-of-matter infrastructure.

Level 1: Avogadro scaling (extensive ∝ amount)
Level 2: Geometry-dependent transport (R = ρL/A)
Level 3: Cross-module consistency (Wiedemann-Franz, Dulong-Petit)
"""

import math
import pytest

from sigma_ground.field.constants import N_AVOGADRO, R_GAS, K_B
from sigma_ground.field.interface.sample import MaterialSample
from sigma_ground.field.interface.surface import MATERIALS


# ── Construction tests ────────────────────────────────────────────────

class TestConstruction:

    def test_from_Z_default_one_mol(self):
        s = MaterialSample.from_Z(29)
        assert abs(s.n_mol - 1.0) < 1e-10

    def test_from_Z_mass(self):
        """63.5g of copper ≈ 1 mol."""
        from sigma_ground.field.interface.element import atomic_mass_kg
        m_atom = atomic_mass_kg(29)
        one_mol_mass = N_AVOGADRO * m_atom
        s = MaterialSample.from_Z(29, mass_kg=one_mol_mass)
        assert abs(s.n_mol - 1.0) < 1e-6

    def test_from_material(self):
        s = MaterialSample.from_material('copper', n_mol=1.0)
        assert s.material_key == 'copper'
        assert s.Z == 29
        assert s.mass_kg > 0
        assert s.volume_m3 > 0

    def test_from_Z_and_mass_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            MaterialSample.from_Z(29, n_mol=1.0, mass_kg=0.065)

    def test_non_integer_Z_rejected(self):
        with pytest.raises(ValueError, match="non-integer"):
            MaterialSample.from_material('rubber', n_mol=1.0)

    def test_repr(self):
        s = MaterialSample.from_material('copper', n_mol=1.0)
        r = repr(s)
        assert 'copper' in r
        assert 'n_mol' in r


# ── Level 1: Avogadro scaling ────────────────────────────────────────

class TestAvogadroScaling:

    def test_double_amount_double_capacity(self):
        """2 mol should have exactly 2× the heat capacity of 1 mol."""
        s1 = MaterialSample.from_material('copper', n_mol=1.0)
        s2 = MaterialSample.from_material('copper', n_mol=2.0)
        C1 = s1.total_heat_capacity_J_K(300)
        C2 = s2.total_heat_capacity_J_K(300)
        assert abs(C2 / C1 - 2.0) < 1e-10

    def test_double_mass_double_capacity(self):
        """Doubling mass doubles heat capacity (by construction)."""
        s1 = MaterialSample.from_material('aluminum', n_mol=1.0)
        s2 = MaterialSample.from_material('aluminum', n_mol=3.0)
        ratio = s2.total_heat_capacity_J_K(500) / s1.total_heat_capacity_J_K(500)
        assert abs(ratio - 3.0) < 1e-10

    def test_phonon_mode_count_one_mol(self):
        """1 mol should have 3 × N_A phonon modes."""
        s = MaterialSample.from_material('aluminum', n_mol=1.0)
        expected = 3 * int(round(N_AVOGADRO))
        assert s.phonon_mode_count() == expected

    def test_phonon_modes_scale_with_amount(self):
        """2 mol → 2× as many phonon modes."""
        s1 = MaterialSample.from_material('iron', n_mol=1.0)
        s2 = MaterialSample.from_material('iron', n_mol=2.0)
        assert s2.phonon_mode_count() == 2 * s1.phonon_mode_count()

    def test_volume_scales_with_moles(self):
        """Volume ∝ amount at constant density."""
        s1 = MaterialSample.from_material('gold', n_mol=1.0)
        s2 = MaterialSample.from_material('gold', n_mol=5.0)
        assert abs(s2.volume_m3 / s1.volume_m3 - 5.0) < 1e-10

    def test_mass_scales_with_moles(self):
        s1 = MaterialSample.from_material('iron', n_mol=1.0)
        s2 = MaterialSample.from_material('iron', n_mol=4.0)
        assert abs(s2.mass_kg / s1.mass_kg - 4.0) < 1e-10


# ── Level 2: Geometry-dependent transport ─────────────────────────────

class TestGeometryTransport:

    def test_copper_wire_resistance_handbook(self):
        """1m of 1mm² copper wire at 300K: R ≈ 17 mΩ (handbook)."""
        s = MaterialSample.from_material('copper', n_mol=1.0)
        R = s.resistance_ohm(length_m=1.0, area_m2=1e-6, T=300.0)
        assert 0.010 < R < 0.025  # 10–25 mΩ

    def test_longer_wire_more_resistance(self):
        """Doubling wire length doubles resistance."""
        s = MaterialSample.from_material('copper', n_mol=1.0)
        R1 = s.resistance_ohm(1.0, 1e-6, 300.0)
        R2 = s.resistance_ohm(2.0, 1e-6, 300.0)
        assert abs(R2 / R1 - 2.0) < 1e-10

    def test_thicker_wire_less_resistance(self):
        """Doubling cross-section halves resistance."""
        s = MaterialSample.from_material('aluminum', n_mol=1.0)
        R1 = s.resistance_ohm(1.0, 1e-6, 300.0)
        R2 = s.resistance_ohm(1.0, 2e-6, 300.0)
        assert abs(R1 / R2 - 2.0) < 1e-10

    def test_aluminum_wire_resistance(self):
        """1m of 1mm² aluminum wire at 300K: R ≈ 27 mΩ (handbook ≈ 26.5 mΩ)."""
        s = MaterialSample.from_material('aluminum', n_mol=1.0)
        R = s.resistance_ohm(1.0, 1e-6, 300.0)
        assert 0.015 < R < 0.040

    def test_iron_wire_resistance(self):
        """1m of 1mm² iron wire at 300K: R ≈ 100 mΩ (handbook ≈ 97 mΩ)."""
        s = MaterialSample.from_material('iron', n_mol=1.0)
        R = s.resistance_ohm(1.0, 1e-6, 300.0)
        assert 0.05 < R < 0.20


# ── Level 3: Cross-module consistency ─────────────────────────────────

class TestCrossModule:

    def test_dulong_petit_high_T(self):
        """1 mol of metal at high T: C ≈ 3R = 24.94 J/(mol·K).

        Dulong-Petit law. Allow 30% tolerance since real metals deviate
        due to electronic contributions and anharmonicity.
        """
        three_R = 3.0 * R_GAS  # 24.94 J/(mol·K)
        for key in ['copper', 'aluminum', 'iron', 'gold', 'nickel']:
            s = MaterialSample.from_material(key, n_mol=1.0)
            C = s.total_heat_capacity_J_K(T=800.0)
            # Allow 30% — Debye model at finite T, electronic corrections
            assert abs(C - three_R) / three_R < 0.30, (
                f"{key}: C={C:.2f} J/(mol·K) vs 3R={three_R:.2f}")

    def test_wiedemann_franz(self):
        """Wiedemann-Franz law: κ/(σT) ≈ L₀ = 2.44×10⁻⁸ W·Ω/K².

        Tests that thermal and electronic transport are consistent.
        Allow 50% — the law is approximate for real metals.
        """
        from sigma_ground.field.interface.thermal import thermal_conductivity
        from sigma_ground.field.interface.electronics import resistivity

        L0 = 2.44e-8  # Lorenz number W·Ω/K²
        T = 300.0
        for key in ['copper', 'aluminum', 'gold']:
            kappa = thermal_conductivity(key, T)   # W/(m·K)
            rho = resistivity(key, T)               # Ω·m
            sigma_e = 1.0 / rho                     # S/m
            L = kappa / (sigma_e * T)
            assert abs(L - L0) / L0 < 0.50, (
                f"{key}: L={L:.3e} vs L₀={L0:.3e}")

    def test_force_to_compress(self):
        """Compression force = K × strain × area.

        1 cm² face, 1% strain on copper.
        K_Cu ≈ 137 GPa → F ≈ 137e9 × 0.01 × 1e-4 ≈ 137 kN.
        """
        s = MaterialSample.from_material('copper', n_mol=1.0)
        F = s.force_to_compress_N(strain=0.01, area_m2=1e-4)
        # K derived from first principles, allow 50% tolerance
        assert 50e3 < F < 300e3, f"F={F:.0f} N"

    def test_energy_to_heat(self):
        """Energy to heat 1 mol Cu from 300K to 400K ≈ C × ΔT.

        With C ≈ 24 J/(mol·K), expect ~2400 J.
        """
        s = MaterialSample.from_material('copper', n_mol=1.0)
        Q = s.energy_to_heat_J(300.0, 400.0)
        assert 1500 < Q < 4000, f"Q={Q:.0f} J"


# ── All metals in MATERIALS ───────────────────────────────────────────

class TestAllMetals:
    """Golden Rule 9: if one, then all."""

    METAL_KEYS = [k for k, v in MATERIALS.items()
                  if v.get('material_type') == 'metal' and isinstance(v.get('Z'), int)]

    def test_all_metals_constructible(self):
        """Every metal in MATERIALS should produce a valid MaterialSample."""
        for key in self.METAL_KEYS:
            s = MaterialSample.from_material(key, n_mol=1.0)
            assert s.mass_kg > 0
            assert s.volume_m3 > 0
            assert s.n_atoms > 0

    def test_all_metals_have_heat_capacity(self):
        """Every metal sample should return finite positive heat capacity."""
        for key in self.METAL_KEYS:
            s = MaterialSample.from_material(key, n_mol=1.0)
            C = s.total_heat_capacity_J_K(300.0)
            assert C > 0 and math.isfinite(C), (
                f"{key}: C={C}")

    def test_all_metals_have_resistance(self):
        """Every metal sample should return finite positive resistance."""
        for key in self.METAL_KEYS:
            s = MaterialSample.from_material(key, n_mol=1.0)
            R = s.resistance_ohm(1.0, 1e-6, 300.0)
            assert R > 0 and math.isfinite(R), (
                f"{key}: R={R}")
