"""
MaterialSample — a physical sample of material with explicit amount and geometry.

Golden Rule 10: collective phenomena must be tested against a volume of matter.
This class wraps existing intensive-property functions with Avogadro scaling
to produce extensive properties on explicit volumes.

Construction:
    MaterialSample.from_Z(Z, n_mol=1.0)           # 1 mol from atomic number
    MaterialSample.from_Z(Z, mass_kg=0.063)        # 63g of copper
    MaterialSample.from_material('copper', n_mol=1) # 1 mol from material key

The sample does NOT recompute physics.  It multiplies intensive properties
by extensive amounts.  The physics lives in the interface modules; the sample
provides the volume context.
"""

import math

from ..constants import N_AVOGADRO, K_B, R_GAS
from .element import (element_properties, atomic_mass_kg,
                      predict_density_kg_m3, predict_crystal_structure)
from .surface import MATERIALS

# ---------------------------------------------------------------------------
# Reverse lookup: Z → material key (for metals in MATERIALS)
# ---------------------------------------------------------------------------
_Z_TO_KEY = {}
for _k, _v in MATERIALS.items():
    _z = _v.get('Z')
    if isinstance(_z, int) and _v.get('material_type') == 'metal':
        if _z not in _Z_TO_KEY:          # first wins (avoid steel overwriting iron)
            _Z_TO_KEY[_z] = _k


class MaterialSample:
    """A physical sample of material with explicit amount and geometry.

    Wraps existing interface functions (thermal, electronics, mechanical,
    superconductivity) so that collective properties can be tested as
    extensive quantities on a real volume of matter.
    """

    def __init__(self, Z, n_atoms, mass_kg, volume_m3, density_kg_m3,
                 material_key=None):
        self.Z = Z
        self.n_atoms = n_atoms
        self.n_mol = n_atoms / N_AVOGADRO
        self.mass_kg = mass_kg
        self.volume_m3 = volume_m3
        self.density_kg_m3 = density_kg_m3
        self.material_key = material_key

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def from_Z(cls, Z, n_mol=None, mass_kg=None):
        """Build a sample from atomic number.

        Provide exactly one of n_mol or mass_kg.  Defaults to 1 mol.
        """
        m_atom = atomic_mass_kg(Z)
        rho = predict_density_kg_m3(Z)
        key = _Z_TO_KEY.get(Z)

        if mass_kg is not None and n_mol is not None:
            raise ValueError("Provide n_mol or mass_kg, not both")
        if mass_kg is not None:
            n_atoms = mass_kg / m_atom
        else:
            if n_mol is None:
                n_mol = 1.0
            n_atoms = n_mol * N_AVOGADRO

        total_mass = n_atoms * m_atom
        volume = total_mass / rho
        return cls(Z, n_atoms, total_mass, volume, rho,
                   material_key=key)

    @classmethod
    def from_material(cls, material_key, n_mol=None, mass_kg=None):
        """Build a sample from a MATERIALS key (e.g. 'copper')."""
        mat = MATERIALS[material_key]
        Z = mat['Z']
        if not isinstance(Z, int):
            raise ValueError(
                f"'{material_key}' has non-integer Z={Z}; "
                "use from_Z for elemental metals")
        rho = mat['density_kg_m3']
        m_atom = atomic_mass_kg(Z)

        if mass_kg is not None and n_mol is not None:
            raise ValueError("Provide n_mol or mass_kg, not both")
        if mass_kg is not None:
            n_atoms = mass_kg / m_atom
        else:
            if n_mol is None:
                n_mol = 1.0
            n_atoms = n_mol * N_AVOGADRO

        total_mass = n_atoms * m_atom
        volume = total_mass / rho
        return cls(Z, n_atoms, total_mass, volume, rho,
                   material_key=material_key)

    # ── Thermal ───────────────────────────────────────────────────────

    def total_heat_capacity_J_K(self, T=300.0):
        """Extensive heat capacity C (J/K) for this sample.

        C = c_p × mass_kg  where c_p is the intensive specific heat.
        """
        from .thermal import specific_heat_j_kg_K
        key = self._require_key()
        c_p = specific_heat_j_kg_K(key, T=T)
        return c_p * self.mass_kg

    def energy_to_heat_J(self, T1, T2, steps=100):
        """Energy (J) needed to heat sample from T1 to T2.

        Integrates C(T) dT using the trapezoidal rule.
        """
        dT = (T2 - T1) / steps
        total = 0.0
        for i in range(steps):
            Ta = T1 + i * dT
            Tb = Ta + dT
            Ca = self.total_heat_capacity_J_K(Ta)
            Cb = self.total_heat_capacity_J_K(Tb)
            total += 0.5 * (Ca + Cb) * dT
        return total

    # ── Phonons ───────────────────────────────────────────────────────

    def phonon_mode_count(self):
        """Total phonon modes = 3 × N_atoms (Dulong-Petit degrees of freedom)."""
        return 3 * int(round(self.n_atoms))

    # ── Electronics ───────────────────────────────────────────────────

    def resistance_ohm(self, length_m, area_m2, T=300.0):
        """Resistance R (Ω) of a wire with given length and cross-section.

        R = ρ × L / A  where ρ is the intensive resistivity.
        """
        from .electronics import resistivity
        key = self._require_key()
        rho_e = resistivity(key, T)
        return rho_e * length_m / area_m2

    # ── Mechanical ────────────────────────────────────────────────────

    def force_to_compress_N(self, strain, area_m2):
        """Force (N) to produce a given compressive strain on a face.

        F = K × strain × A  where K is the bulk modulus.
        """
        from .mechanical import bulk_modulus
        key = self._require_key()
        K = bulk_modulus(key)
        return K * strain * area_m2

    # ── Superconductivity ─────────────────────────────────────────────

    def cooling_profile(self, T_start, T_end, steps=50):
        """Simulate cooling this sample through its superconducting transition.

        Wraps block_cooling_profile with the sample's material key.
        """
        from .superconductivity import block_cooling_profile
        key = self._require_key()
        # Map material key to superconductor key
        sc_key = MATERIALS[key]['name'].lower()
        return block_cooling_profile(sc_key, T_start, T_end, steps)

    # ── Helpers ───────────────────────────────────────────────────────

    def _require_key(self):
        """Return material_key or raise if not available."""
        if self.material_key is None:
            raise ValueError(
                f"No material key for Z={self.Z}. "
                "Use from_material() or a Z that maps to MATERIALS.")
        return self.material_key

    def __repr__(self):
        key = self.material_key or f'Z={self.Z}'
        return (f"MaterialSample({key}, "
                f"n_mol={self.n_mol:.4g}, "
                f"mass={self.mass_kg:.4g} kg, "
                f"vol={self.volume_m3:.4g} m³)")
