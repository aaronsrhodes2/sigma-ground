"""
Dielectric properties — permittivity from electronic structure.

Derivation chain:
  electronics.py (free electron density, plasma frequency)
  + optics.py (Drude permittivity, Cauchy refractive index)
  → dielectric.py (static and frequency-dependent ε_r)

The dielectric constant ε_r is NOT an independent number to look up.
For metals: it's derived from free electron density (Drude).
For insulators: it's derived from refractive index (Maxwell relation)
or from polarizability (Clausius-Mossotti).

Derivation chains:

  1. Metal: Drude Static Permittivity (FIRST_PRINCIPLES)
     ε_r(ω=0) = 1 − ωp² / γ²   (for DC, ω → 0)

     But this gives a large negative number (metals screen E-fields).
     The physically meaningful quantity is ωp and γ, which determine
     the frequency-dependent response.

  2. Insulator: Maxwell Relation (FIRST_PRINCIPLES)
     ε_r = n²   (at optical frequencies, for non-magnetic materials)

     Where n is the refractive index from Cauchy coefficients.
     This is exact at optical frequencies where μ_r = 1.

  3. Clausius-Mossotti (FIRST_PRINCIPLES)
     (ε_r − 1)/(ε_r + 2) = N × α / (3ε₀)

     Where:
       N = number density of molecules (1/m³)
       α = molecular polarizability (F·m²)

     For known polarizabilities, this gives ε_r from first principles.

  4. Frequency Dependence (FIRST_PRINCIPLES: Drude/Lorentz)
     Metals: ε(ω) = 1 − ωp²/(ω² + iγω)   (Drude)
     Insulators: ε(ω) ≈ n(ω)²              (Maxwell, valid below bandgap)

  5. Dielectric Loss (FIRST_PRINCIPLES)
     tan δ = ε"/ε'  (ratio of imaginary to real permittivity)

     For metals: dominated by free-electron Drude loss
     For insulators: dominated by phonon absorption and defect losses

  6. Dielectric Breakdown (FIRST_PRINCIPLES form + MEASURED calibration)
     E_b ≈ C × E_g / (e × d_lattice)

     Where:
       E_g = bandgap energy (eV) — wider gap → harder to ionize
       d_lattice = lattice spacing (m) — longer MFP → more avalanche gain
       C = calibration constant (~0.1, MEASURED)

     Avalanche breakdown: an electron accelerated across one lattice
     spacing must gain enough energy to ionize another electron.
     E_b = E_g / (e × λ_mfp) where λ_mfp ∝ d_lattice.

σ-dependence:
  ε_r for insulators: depends on n², which is EM → σ-INVARIANT.
  ε_r for metals: depends on ωp (electron density) → σ-INVARIANT
  (electron count is set by Z, not by nuclear mass).
  Breakdown field: E_g is EM → invariant. Lattice spacing shifts
  weakly with σ through nuclear mass → very weak σ-dependence.

  Dielectric properties are effectively σ-INVARIANT.

Origin tags:
  - Drude permittivity: FIRST_PRINCIPLES (free-electron EM)
  - Maxwell relation: FIRST_PRINCIPLES (Maxwell's equations)
  - Clausius-Mossotti: FIRST_PRINCIPLES (mean-field polarization)
  - Breakdown field: FIRST_PRINCIPLES form + MEASURED C
"""

import math
from ..constants import EPS_0, E_CHARGE, SIGMA_HERE


# ── Derived helpers ──────────────────────────────────────────────
# eps_inf = n² (Maxwell relation, FIRST_PRINCIPLES)
# Where n comes from Cauchy coefficients in optics.py.
# bandgap from Varshni equation in semiconductor_optics.py.

def _eps_inf_from_cauchy(optics_key):
    """ε_∞ = n² — DERIVED from Cauchy refractive index (Maxwell relation).

    At long wavelengths, n → A (Cauchy's first coefficient).
    ε_∞ = A² is exact for non-magnetic materials (μ_r = 1).
    """
    from .optics import CAUCHY_COEFFICIENTS
    A, _B = CAUCHY_COEFFICIENTS[optics_key]
    return A * A


def _bandgap_from_varshni(sc_key, T=300.0):
    """Bandgap — DERIVED from Varshni equation at temperature T.

    E_g(T) = E_g0 - α T² / (T + β)
    """
    from .semiconductor_optics import band_gap_ev
    return band_gap_ev(sc_key, T)


# ── Dielectric Data ───────────────────────────────────────────────
# Static dielectric constants: MEASURED where available, DERIVED
# from refractive index or Clausius-Mossotti where not.
#
# For metals: ε_r is not meaningful at DC (perfect screening).
# We store the optical-frequency ε_∞ from the Drude model instead.
#
# eps_inf for insulators: DERIVED from n² (Maxwell relation)
#   via optics.py Cauchy coefficients where available.
# bandgap_eV: DERIVED from Varshni equation (semiconductor_optics.py)
#   at 300K where available.
#
# Sources: CRC Handbook, Kittel "Solid State Physics",
#          Palik "Handbook of Optical Constants"
#
# Rule 9: every material in the main MATERIALS database.

DIELECTRIC_DATA = {
    # Metals — optical permittivity ε_∞ (high-frequency limit)
    # ε_∞ ≈ 1 for simple metals (interband transitions add ~5-10)
    # These are MEASURED (Palik), not derivable from Cauchy (metals are lossy).
    'iron':     {'type': 'metal', 'eps_inf': 1.0, 'eps_static': None},
    'copper':   {'type': 'metal', 'eps_inf': 6.0, 'eps_static': None},
    'aluminum': {'type': 'metal', 'eps_inf': 1.0, 'eps_static': None},
    'gold':     {'type': 'metal', 'eps_inf': 9.0, 'eps_static': None},
    'nickel':   {'type': 'metal', 'eps_inf': 1.0, 'eps_static': None},
    'tungsten': {'type': 'metal', 'eps_inf': 1.0, 'eps_static': None},
    'titanium': {'type': 'metal', 'eps_inf': 1.0, 'eps_static': None},
    # Semiconductors — eps_inf is MEASURED (interband, not derivable from Cauchy)
    # bandgap_eV: DERIVED from Varshni at 300K (semiconductor_optics.py)
    'silicon':        {'type': 'semiconductor', 'eps_inf': 11.7, 'eps_static': 11.7,
                       'bandgap_eV': _bandgap_from_varshni('silicon'),
                       'lattice_pm': 543},
    'germanium':      {'type': 'semiconductor', 'eps_inf': 16.0, 'eps_static': 16.0,
                       'bandgap_eV': _bandgap_from_varshni('germanium'),
                       'lattice_pm': 566},
    'gallium_arsenide': {'type': 'semiconductor', 'eps_inf': 10.9, 'eps_static': 12.9,
                         'bandgap_eV': _bandgap_from_varshni('gallium_arsenide'),
                         'lattice_pm': 565},
    # Insulators — eps_inf DERIVED from n² (Maxwell relation, optics.py)
    # bandgap_eV: DERIVED from Varshni for diamond; MEASURED for wide-gap insulators
    'silicon_dioxide': {'type': 'insulator',
                        'eps_inf': _eps_inf_from_cauchy('fused_silica'),
                        'eps_static': 3.9,
                        'bandgap_eV': 9.0, 'lattice_pm': 500},  # SiO₂ gap: MEASURED (too wide for Varshni)
    'alumina':         {'type': 'insulator',
                        'eps_inf': _eps_inf_from_cauchy('sapphire'),
                        'eps_static': 9.4,
                        'bandgap_eV': 8.8, 'lattice_pm': 476},  # Al₂O₃ gap: MEASURED
    'diamond':         {'type': 'insulator',
                        'eps_inf': _eps_inf_from_cauchy('diamond'),
                        'eps_static': _eps_inf_from_cauchy('diamond'),  # non-polar: ε_s = ε_∞
                        'bandgap_eV': _bandgap_from_varshni('diamond'),
                        'lattice_pm': 357},
    'barium_titanate': {'type': 'insulator', 'eps_inf': 5.4, 'eps_static': 1700.0,
                        'bandgap_eV': 3.2, 'lattice_pm': 403},  # MEASURED (no Cauchy/Varshni)
    'polyethylene':    {'type': 'insulator', 'eps_inf': 2.3, 'eps_static': 2.3,
                        'bandgap_eV': 8.0, 'lattice_pm': 740},  # MEASURED (polymer)
    'water_25C':       {'type': 'insulator',
                        'eps_inf': _eps_inf_from_cauchy('water'),
                        'eps_static': 78.4,
                        'bandgap_eV': 6.5, 'lattice_pm': 280},
}


# ── Dielectric Constant ──────────────────────────────────────────

def dielectric_constant(material_key, frequency_hz=0.0):
    """Relative permittivity ε_r at given frequency.

    FIRST_PRINCIPLES:
      DC (f=0): static ε_r (MEASURED)
      Optical (f~10¹⁴): ε_∞ = n² (Maxwell relation)
      Intermediate: Debye relaxation model

    For metals at f=0: returns ε_∞ (the physically useful quantity;
    true DC permittivity is −∞ for a perfect conductor).

    For insulators with ionic polarization (ε_static >> ε_∞):
      ε(f) = ε_∞ + (ε_static − ε_∞) / (1 + (f/f_relax)²)
    where f_relax is the ionic relaxation frequency (~10¹² Hz).

    Args:
        material_key: key into DIELECTRIC_DATA
        frequency_hz: frequency in Hz (0 = static)

    Returns:
        Relative permittivity ε_r (dimensionless).
    """
    data = DIELECTRIC_DATA[material_key]

    if data['type'] == 'metal':
        return data['eps_inf']

    eps_static = data['eps_static']
    eps_inf = data['eps_inf']

    if frequency_hz <= 0:
        return eps_static

    # Debye relaxation: ionic polarization freezes out at ~THz
    # f_relax ≈ 10¹² Hz for most ionic solids, ~10¹⁰ for water
    if material_key == 'water_25C':
        f_relax = 18e9  # MEASURED: water Debye relaxation (18 GHz)
    else:
        f_relax = 1e12  # typical ionic relaxation frequency

    ratio = (frequency_hz / f_relax) ** 2
    return eps_inf + (eps_static - eps_inf) / (1.0 + ratio)


def dielectric_loss_tangent(material_key, frequency_hz):
    """Dielectric loss tangent tan δ = ε"/ε'.

    FIRST_PRINCIPLES: Debye relaxation model.

    tan δ = (ε_s − ε_∞) × (f/f_relax) / (ε_∞ + ε_s × (f/f_relax)²)
           × 1/(1 + (f/f_relax)²)

    For metals: tan δ is dominated by conduction loss:
      tan δ = σ_DC / (2π f ε₀ ε_r)
    Not applicable (diverges as f→0).

    Args:
        material_key: key into DIELECTRIC_DATA
        frequency_hz: frequency in Hz

    Returns:
        Loss tangent (dimensionless).
    """
    data = DIELECTRIC_DATA[material_key]

    if data['type'] == 'metal' or frequency_hz <= 0:
        return 0.0

    eps_static = data['eps_static']
    eps_inf = data['eps_inf']

    if material_key == 'water_25C':
        f_relax = 18e9
    else:
        f_relax = 1e12

    x = frequency_hz / f_relax
    delta_eps = eps_static - eps_inf

    eps_prime = eps_inf + delta_eps / (1.0 + x ** 2)
    eps_double_prime = delta_eps * x / (1.0 + x ** 2)

    if eps_prime <= 0:
        return 0.0

    return eps_double_prime / eps_prime


# ── Clausius-Mossotti ─────────────────────────────────────────────

def clausius_mossotti(polarizability_m3, number_density):
    """Dielectric constant from Clausius-Mossotti relation.

    (ε_r − 1)/(ε_r + 2) = N α / (3ε₀)

    FIRST_PRINCIPLES: mean-field theory of induced dipoles.

    Args:
        polarizability_m3: molecular polarizability in m³
            (note: 1 ų = 1.113e-40 F·m², but in CGS, α/4πε₀ in m³)
        number_density: number density N in 1/m³

    Returns:
        Relative permittivity ε_r.
    """
    chi = number_density * polarizability_m3 / (3.0 * EPS_0)

    if chi >= 1.0:
        # Approaching ferroelectric instability
        return 1000.0  # cap at very high ε_r

    # Solve: (ε-1)/(ε+2) = chi → ε = (1 + 2χ)/(1 − χ)
    return (1.0 + 2.0 * chi) / (1.0 - chi)


# ── Dielectric Breakdown ──────────────────────────────────────────

def breakdown_field(material_key):
    """Dielectric breakdown field strength (V/m).

    FIRST_PRINCIPLES form + MEASURED calibration:
      E_b ≈ 0.1 × E_g / (e × d_lattice)

    An electron accelerated across one mean free path (≈ lattice spacing)
    must gain enough energy to ionize across the bandgap. The 0.1
    factor accounts for the ionization threshold being a fraction of
    E_g (phonon losses, momentum conservation).

    Accuracy: factor of 2-3 (breakdown is very sensitive to defects,
    temperature, sample thickness). This gives the INTRINSIC limit.

    Args:
        material_key: key into DIELECTRIC_DATA

    Returns:
        Breakdown field in V/m, or 0 for metals.
    """
    data = DIELECTRIC_DATA[material_key]

    if data['type'] == 'metal':
        return 0.0  # metals conduct, not break down

    E_g_eV = data.get('bandgap_eV', 0)
    d_pm = data.get('lattice_pm', 500)

    if E_g_eV <= 0 or d_pm <= 0:
        return 0.0

    E_g_J = E_g_eV * E_CHARGE
    d_m = d_pm * 1e-12

    # Avalanche threshold: 0.1 × E_g / (e × d)
    return 0.1 * E_g_J / (E_CHARGE * d_m)


def breakdown_voltage(material_key, thickness_m):
    """Dielectric breakdown voltage for a slab (V).

    V_b = E_b × d

    FIRST_PRINCIPLES: uniform field in parallel-plate geometry.

    Args:
        material_key: key into DIELECTRIC_DATA
        thickness_m: dielectric thickness in metres

    Returns:
        Breakdown voltage in Volts.
    """
    return breakdown_field(material_key) * thickness_m


# ── Capacitor Energy Density ──────────────────────────────────────

def energy_density(material_key, E_field):
    """Electrostatic energy density (J/m³).

    u = ½ ε₀ ε_r E²

    FIRST_PRINCIPLES: electrostatic energy in a linear dielectric.

    Args:
        material_key: key into DIELECTRIC_DATA
        E_field: electric field strength (V/m)

    Returns:
        Energy density in J/m³.
    """
    eps_r = dielectric_constant(material_key, 0)
    return 0.5 * EPS_0 * eps_r * E_field ** 2


def max_energy_density(material_key):
    """Maximum storable energy density (J/m³).

    u_max = ½ ε₀ ε_r E_b²

    The practical limit: field at breakdown.

    Args:
        material_key: key into DIELECTRIC_DATA

    Returns:
        Maximum energy density in J/m³.
    """
    E_b = breakdown_field(material_key)
    return energy_density(material_key, E_b)


# ── Diagnostics ───────────────────────────────────────────────────

def dielectric_report(material_key, frequency_hz=0.0):
    """Complete dielectric report."""
    data = DIELECTRIC_DATA[material_key]

    report = {
        'material': material_key,
        'type': data['type'],
        'eps_r_static': dielectric_constant(material_key, 0),
        'eps_r_optical': data['eps_inf'],
    }

    if frequency_hz > 0:
        report['eps_r_at_freq'] = dielectric_constant(material_key, frequency_hz)
        report['frequency_hz'] = frequency_hz
        report['loss_tangent'] = dielectric_loss_tangent(material_key, frequency_hz)

    if data['type'] != 'metal':
        E_b = breakdown_field(material_key)
        report['breakdown_field_V_m'] = E_b
        report['breakdown_field_MV_m'] = E_b / 1e6
        report['max_energy_density_J_m3'] = max_energy_density(material_key)

    return report


def full_report(frequency_hz=0.0):
    """Reports for ALL dielectrics. Rule 9."""
    return {key: dielectric_report(key, frequency_hz)
            for key in DIELECTRIC_DATA}
