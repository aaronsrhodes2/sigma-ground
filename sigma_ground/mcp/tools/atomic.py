"""Atomic physics tools: Rydberg formula, hydrogen-like spectra, ionization.

Pure formulas + curated lookup tables. The Rydberg formula here is
generalized to hydrogen-like ions (He+, Li2+, etc.) by Z^2 scaling.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


# Periodic-table data is now provided by materials.element_atomic_data
# (uses the periodictable library). This local table was removed to
# avoid duplicate registrations. Kept commented for reference.
_UNUSED_PERIODIC_DATA_FOR_REFERENCE: dict[str, tuple[int, float]] = {
    "H":  (1,   1.008),    "He": (2,   4.0026),
    "Li": (3,   6.94),     "Be": (4,   9.0122),
    "B":  (5,  10.81),     "C":  (6,  12.011),
    "N":  (7,  14.007),    "O":  (8,  15.999),
    "F":  (9,  18.998),    "Ne": (10, 20.180),
    "Na": (11, 22.990),    "Mg": (12, 24.305),
    "Al": (13, 26.982),    "Si": (14, 28.085),
    "P":  (15, 30.974),    "S":  (16, 32.06),
    "Cl": (17, 35.45),     "Ar": (18, 39.948),
    "K":  (19, 39.098),    "Ca": (20, 40.078),
    "Sc": (21, 44.956),    "Ti": (22, 47.867),
    "V":  (23, 50.942),    "Cr": (24, 51.996),
    "Mn": (25, 54.938),    "Fe": (26, 55.845),
    "Co": (27, 58.933),    "Ni": (28, 58.693),
    "Cu": (29, 63.546),    "Zn": (30, 65.38),
    "Ga": (31, 69.723),    "Ge": (32, 72.630),
    "As": (33, 74.922),    "Se": (34, 78.971),
    "Br": (35, 79.904),    "Kr": (36, 83.798),
    "Rb": (37, 85.468),    "Sr": (38, 87.62),
    "Y":  (39, 88.906),    "Zr": (40, 91.224),
    "Nb": (41, 92.906),    "Mo": (42, 95.95),
    "Tc": (43, 98.0),      "Ru": (44, 101.07),
    "Rh": (45, 102.91),    "Pd": (46, 106.42),
    "Ag": (47, 107.868),   "Cd": (48, 112.414),
    "In": (49, 114.818),   "Sn": (50, 118.710),
    "Sb": (51, 121.760),   "Te": (52, 127.60),
    "I":  (53, 126.904),   "Xe": (54, 131.293),
    "Cs": (55, 132.905),   "Ba": (56, 137.327),
    "La": (57, 138.905),   "Ce": (58, 140.116),
    "Pr": (59, 140.908),   "Nd": (60, 144.242),
    "W":  (74, 183.84),    "Pt": (78, 195.084),
    "Au": (79, 196.967),   "Hg": (80, 200.592),
    "Pb": (82, 207.2),     "Bi": (83, 208.980),
    "Th": (90, 232.038),   "U":  (92, 238.02891),
    "Pu": (94, 244.0),
}


# element_atomic_data() moved to sigma_ground.mcp.tools.materials
# (it uses the periodictable library for complete coverage). Don't
# re-define here -- would create a duplicate MCP registration.


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

    Accepts symbol ('H', 'Fe', 'Hg'), full name ('hydrogen', 'iron',
    'mercury_element' to disambiguate from the planet), Latin name
    ('ferrum', 'aurum'), or atomic number ('1', 'Z=1', '26').
    """
    from sigma_ground.mcp.tools.aliases import ELEMENT_ALIASES, resolve
    resolved = resolve(element_symbol, ELEMENT_ALIASES)
    # If the alias map returned a symbol (length 1-2, starts with capital),
    # use it. Otherwise try the input's capitalize() as a last resort.
    sym = resolved if resolved in _FIRST_IONIZATION_EV else element_symbol.strip().capitalize()
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


_PARTICLE_REST_MASS_KG = {
    "electron": 9.1093837015e-31,
    "proton":   1.67262192369e-27,
    "neutron":  1.67492749804e-27,
    "alpha":    6.6446573357e-27,   # helium-4 nucleus
    "muon":     1.883531627e-28,
}


def de_broglie_from_kinetic_energy(kinetic_energy_eV: float,
                                     particle: str = "electron"
                                     ) -> ToolResult:
    """de Broglie wavelength from KINETIC ENERGY (not velocity).

    Collapses the chain the benchmark kept failing: a "1 keV electron"
    question needs KE -> momentum -> wavelength. This does it in one call,
    relativistically exact so it's correct at any energy:
        E_total = KE + m c²,   p = sqrt(E_total² − (m c²)²)/c,   λ = h/p

    Parameters
    ----------
    kinetic_energy_eV : float
        Kinetic energy in electronvolts (1 keV = 1000, 1 MeV = 1e6).
    particle : str
        'electron' (default), 'proton', 'neutron', 'alpha', 'muon'.

    Example:
      de_broglie_from_kinetic_energy(1000, "electron")  # 1 keV e- -> 3.88e-11 m
    """
    p_key = (particle or "electron").strip().lower()
    m = _PARTICLE_REST_MASS_KG.get(p_key)
    if m is None:
        return ToolResult(value=None, source="unknown particle",
                           notes=f"Known: {sorted(_PARTICLE_REST_MASS_KG)}",
                           inputs={"particle": particle})
    if kinetic_energy_eV <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"kinetic_energy_eV": kinetic_energy_eV})
    from sigma_ground.field.constants import HBAR, C
    eV_J = 1.602176634e-19
    h = HBAR * 2.0 * math.pi
    KE = kinetic_energy_eV * eV_J
    mc2 = m * C * C
    E_total = KE + mc2
    p = math.sqrt(max(E_total * E_total - mc2 * mc2, 0.0)) / C
    lam = h / p
    rel = "relativistic" if KE > 0.01 * mc2 else "non-relativistic regime"
    return ToolResult(
        value=lam, units="m",
        source="sigma-ground (de Broglie, relativistic momentum)",
        formula="λ = h c / sqrt(KE² + 2 KE m c²)",
        inputs={"kinetic_energy_eV": kinetic_energy_eV, "particle": p_key,
                "rest_mass_kg": m},
        notes=(f"{p_key} at {kinetic_energy_eV:g} eV ({rel}). "
                f"Exact at all energies; reduces to h/sqrt(2 m KE) "
                f"non-relativistically."),
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
