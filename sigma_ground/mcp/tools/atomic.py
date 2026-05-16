"""Atomic physics tools: Rydberg formula, hydrogen-like spectra, ionization.

Pure formulas + curated lookup tables. The Rydberg formula here is
generalized to hydrogen-like ions (He+, Li2+, etc.) by Z^2 scaling.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


# First ionization energies in eV. NIST Atomic Spectra Database.
# https://physics.nist.gov/cgi-bin/ASD/ie.pl
_FIRST_IONIZATION_EV: dict[str, tuple[float, str]] = {
    "H":  (13.598,  "NIST ASD (Hydrogen, ground state)"),
    "He": (24.587,  "NIST ASD (Helium first IE)"),
    "Li": (5.392,   "NIST ASD (Lithium first IE)"),
    "Be": (9.323,   "NIST ASD"),
    "B":  (8.298,   "NIST ASD"),
    "C":  (11.260,  "NIST ASD"),
    "N":  (14.534,  "NIST ASD"),
    "O":  (13.618,  "NIST ASD"),
    "F":  (17.423,  "NIST ASD"),
    "Ne": (21.565,  "NIST ASD"),
    "Na": (5.139,   "NIST ASD"),
    "Mg": (7.646,   "NIST ASD"),
    "Al": (5.986,   "NIST ASD"),
    "Si": (8.152,   "NIST ASD"),
    "P":  (10.487,  "NIST ASD"),
    "S":  (10.360,  "NIST ASD"),
    "Cl": (12.968,  "NIST ASD"),
    "Ar": (15.760,  "NIST ASD"),
    "K":  (4.341,   "NIST ASD"),
    "Ca": (6.113,   "NIST ASD"),
    "Fe": (7.902,   "NIST ASD"),
    "Cu": (7.726,   "NIST ASD"),
    "Au": (9.226,   "NIST ASD"),
    "Hg": (10.438,  "NIST ASD"),
    "U":  (6.194,   "NIST ASD"),
}


def first_ionization_energy(element_symbol: str) -> ToolResult:
    """First ionization energy in eV for an element.

    NIST Atomic Spectra Database. The first IE is the energy to remove
    one electron from the ground-state neutral atom.
    """
    sym = element_symbol.strip().capitalize()
    if sym not in _FIRST_IONIZATION_EV:
        return ToolResult(
            value=None,
            source="sigma-ground (ionization energy lookup)",
            notes=(f"First IE not in lookup for '{element_symbol}'. "
                    f"Available: {sorted(_FIRST_IONIZATION_EV.keys())[:20]}..."),
            inputs={"element_symbol": element_symbol},
        )
    value, src = _FIRST_IONIZATION_EV[sym]
    return ToolResult(
        value=value,
        units="eV",
        source=f"sigma-ground via {src}",
        provenance_tag="VERIFIED",
        inputs={"element_symbol": element_symbol},
        notes=("First ionization energy: minimum energy to remove the "
                "outermost electron from the neutral atom. Larger IE -> "
                "less reactive (noble gases peak)."),
    )


def hydrogen_like_energy_level(n: int, atomic_number: int = 1) -> ToolResult:
    """Energy of level n in a hydrogen-like ion: E_n = -13.606 Z^2 / n^2 eV.

    Parameters
    ----------
    n : int
        Principal quantum number, n >= 1.
    atomic_number : int
        Z. For neutral H, Z=1. For He+, Z=2. For Li2+, Z=3.
    """
    if not isinstance(n, int) or n < 1:
        return ToolResult(value=None, source="invalid input",
                           notes="n must be a positive integer",
                           inputs={"n": n, "atomic_number": atomic_number})
    if not isinstance(atomic_number, int) or atomic_number < 1:
        return ToolResult(value=None, source="invalid input",
                           notes="atomic_number must be a positive integer",
                           inputs={"n": n, "atomic_number": atomic_number})
    # Use Rydberg energy in eV: 13.605693122994 eV (CODATA 2018)
    Ry_eV = 13.605693122994
    E_n = -Ry_eV * atomic_number ** 2 / n ** 2
    return ToolResult(
        value=E_n,
        units="eV",
        source="sigma-ground (Bohr model / Schrodinger hydrogen-like)",
        formula="E_n = -Ry Z^2 / n^2",
        inputs={"n": n, "atomic_number": atomic_number},
        notes=("Ground state of hydrogen (n=1, Z=1): -13.606 eV. "
                "Bound states are negative. Ionization energy from level n "
                "is |E_n|."),
    )


def hydrogen_emission_wavelength(n_initial: int, n_final: int,
                                    atomic_number: int = 1) -> ToolResult:
    """Wavelength of transition n_initial -> n_final in hydrogen-like atom.

    Uses Rydberg formula with Z^2 scaling:
    1/lambda = R_inf Z^2 (1/n_f^2 - 1/n_i^2)

    For n_i > n_f, this is emission (positive wavelength).
    """
    if (not isinstance(n_initial, int) or not isinstance(n_final, int)
            or not isinstance(atomic_number, int)):
        return ToolResult(value=None, source="invalid input",
                           notes="all quantum numbers must be integers",
                           inputs={"n_initial": n_initial,
                                   "n_final": n_final,
                                   "atomic_number": atomic_number})
    if n_initial < 1 or n_final < 1 or n_initial == n_final or atomic_number < 1:
        return ToolResult(value=None, source="invalid input",
                           inputs={"n_initial": n_initial,
                                   "n_final": n_final,
                                   "atomic_number": atomic_number})
    R_inf = 1.0973731568160e7  # 1/m, CODATA 2018
    inv_lambda = R_inf * atomic_number ** 2 * abs(
        1.0/n_final**2 - 1.0/n_initial**2)
    lam = 1.0 / inv_lambda
    series_map = {1: "Lyman (UV)", 2: "Balmer (visible)",
                   3: "Paschen (IR)", 4: "Brackett (IR)", 5: "Pfund (IR)"}
    n_low = min(n_initial, n_final)
    series = series_map.get(n_low, "higher series (deep IR/radio)")
    return ToolResult(
        value=lam,
        units="m",
        source="sigma-ground (Rydberg formula, CODATA 2018 R_inf)",
        formula="1/lambda = R_inf Z^2 |1/n_f^2 - 1/n_i^2|",
        inputs={"n_initial": n_initial, "n_final": n_final,
                "atomic_number": atomic_number},
        notes=(f"{series} series for Z={atomic_number}. "
                f"Notable H lines: H-alpha (3->2)=656 nm, "
                f"Lyman-alpha (2->1)=121.6 nm."),
    )


def de_broglie_wavelength(mass_kg: float, velocity_m_s: float) -> ToolResult:
    """de Broglie matter wave: lambda = h / (m v).

    Non-relativistic; valid when v << c.
    """
    if mass_kg <= 0 or velocity_m_s <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg,
                                   "velocity_m_s": velocity_m_s})
    from sigma_ground.field.constants import HBAR
    h = HBAR * 2.0 * 3.14159265358979323846
    lam = h / (mass_kg * velocity_m_s)
    return ToolResult(
        value=lam,
        units="m",
        source="sigma-ground (de Broglie 1924)",
        formula="lambda = h / (m v)",
        inputs={"mass_kg": mass_kg, "velocity_m_s": velocity_m_s},
        notes=("Matter wave. Electron at 1 keV: ~0.04 nm (X-ray scale). "
                "Proton at thermal speed (300 K, ~2500 m/s): ~80 nm. "
                "Macroscopic objects: << atomic scale, no diffraction."),
    )


def photon_energy_from_wavelength(wavelength_m: float) -> ToolResult:
    """E = h c / lambda. Photon energy from wavelength."""
    if wavelength_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"wavelength_m": wavelength_m})
    from sigma_ground.field.constants import HBAR, C
    h = HBAR * 2.0 * 3.14159265358979323846
    E = h * C / wavelength_m
    # Also report in eV for convenience
    eV = E / 1.602176634e-19
    return ToolResult(
        value=E,
        units="J",
        source="sigma-ground (Planck-Einstein relation)",
        formula="E = h c / lambda",
        inputs={"wavelength_m": wavelength_m},
        notes=(f"= {eV:.4g} eV. Visible: ~1.6-3.3 eV. UV starts at ~3.3 eV. "
                f"X-rays: 100 eV to 100 keV. Gamma: > 100 keV."),
    )


def photon_energy_from_frequency(frequency_hz: float) -> ToolResult:
    """E = h f. Photon energy from frequency."""
    if frequency_hz <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"frequency_hz": frequency_hz})
    from sigma_ground.field.constants import HBAR
    h = HBAR * 2.0 * 3.14159265358979323846
    E = h * frequency_hz
    eV = E / 1.602176634e-19
    return ToolResult(
        value=E,
        units="J",
        source="sigma-ground (Planck-Einstein relation)",
        formula="E = h f",
        inputs={"frequency_hz": frequency_hz},
        notes=f"= {eV:.4g} eV.",
    )
